from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer
from rest_framework.permissions import IsAuthenticated
from .llm_gpt import (short_term_memory_llm, long_term_memory_llm, 
                     check_gpt_knows_answer_yes_no)
from pymilvus import MilvusClient
from chat.utils import store_message_milvus
from datetime import datetime
from django.conf import settings
from pymilvus import model




class ConversationView(APIView):
    
    permission_classes = [IsAuthenticated]
    serializer_class = ConversationSerializer
    model_class = Conversation


    def post(self, request, *args, **kwargs):
        request.data['user'] = request.user.id
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, *args, **kwargs):
        serializer = self.serializer_class(
            self.model_class.objects.filter(user=request.user).order_by('-created_at'),
            many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def patch(self, request, *args, **kwargs):
        conversation_id =  request.data.get('conversation_id', 0)
        conversation = self.model_class.objects.filter(id=conversation_id, user=request.user).first()
        
        if conversation is None:
            return Response({'detail': 'Conversation not found'}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = self.serializer_class(conversation, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, *args, **kwargs):
        conversation_id =  request.data.get('conversation_id', 0)
        conversation = self.model_class.objects.filter(id=conversation_id, user=request.user).first()

        if conversation is None:
            return Response({'detail': 'Conversation not found'}, status=status.HTTP_404_NOT_FOUND)

        conversation.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)



class MessageView(APIView):
    
    permission_classes = [IsAuthenticated]
    model_class = Conversation
    client = MilvusClient(settings.MILVUS_DB_NAME)
    

    def post(self, request, *args, **kwargs):
        conversation_id =  request.data.get('conversation_id', 0)
        conversation = self.model_class.objects.filter(id=conversation_id, user=request.user).first()
        
        if conversation is None:
            return Response({'detail': 'Conversation not found'}, status=status.HTTP_404_NOT_FOUND)

        prompt = request.data.get("prompt")
        response = short_term_memory_llm(prompt, conversation_id)
        
        if not check_gpt_knows_answer_yes_no(prompt, response):
            # Check if the gpt does not find any information from the previous context.
            # Then just get the previous all chats from milvous as context and then query the gpt.
            response = long_term_memory_llm(self.client, request.user, prompt, response, conversation_id)

        message = Message(
            conversation=conversation,
            query=prompt,
            response=response
        )
        message.save()
        
        # Saving the last message to get conversations more faster
        conversation.last_message = response
        conversation.save()
        
        embedding_fn = model.DefaultEmbeddingFunction()
        vectors = embedding_fn.encode_documents([prompt +". "+ response ])

        store_message_milvus(
            client=self.client,
            collection_name="message",
            data=[{
                    "id": message.id,
                    "user_id": request.user.id,
                    "conversation_id": conversation_id,
                    "vector": vectors[0],
                    "prompt": prompt,
                    "response": response,
                    "timestamp": str(datetime.now())
                }])

        return Response(
            MessageSerializer(message).data,
            status=status.HTTP_200_OK
        )

    def get(self, request, *args, **kwargs):
        conversation_id =  request.query_params.get('conversation_id', 0)
        conversation = self.model_class.objects.filter(id=conversation_id, user=request.user).first()
        
        if conversation is None:
            return Response({'detail': 'Conversation not found'}, status=status.HTTP_404_NOT_FOUND)
        
        messages = Message.objects.filter(conversation__id=conversation_id)
        serializer = MessageSerializer(messages, many=True)

        return Response(
            serializer.data,
            status=status.HTTP_200_OK
        )


class Milvus(APIView):
    
    client = MilvusClient(settings.MILVUS_DB_NAME)

    def post(self, request, *args, **kwargs):
        if not request.data.get("collection_name"):
            return Response(
                {
                    "error": "No collection name provided"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        self.client.create_collection(
            collection_name=request.data.get("collection_name"),
            dimension=768,
        )
        return Response(
            {
                "message": "Collection created successfully"
            },
            status=status.HTTP_200_OK
        )
    
    def get(self, request, *args, **kwargs):
        # get all data from each collection based on search.
        result = []
        # client = MilvusClient(settings.MILVUS_DB_NAME)
        # client.delete(collection_name="message", ids=[246])
        embedding_fn = model.DefaultEmbeddingFunction()
        query = request.query_params.get("query", "tell me AI related information")

        for collection in self.client.list_collections():
            data = self.client.search(
                collection_name=collection,
                  filter=f"user_id == {int(request.user.id)}",
                  data=embedding_fn.encode_queries([query]),
                output_fields=["id", "user_id", "conversation_id",
                                "prompt", "response", "timestamp"],
                limit=100
            )
            
            d = {
                "collection": collection,
                "data": data
            }
            result.append(d)

        return Response(
            result,
            status=status.HTTP_200_OK
        )