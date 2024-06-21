import openai as open_ai



def store_message_milvus(client, collection_name, data):
    client.insert(
            collection_name=collection_name,
            data=data
    )
    
def get_embedding(text, model_id="text-embedding-ada-002"):
        response = open_ai.Embedding.create(
                input=text,
                model=model_id
        )
        embeddings = response['data'][0]['embedding']
        dimensions = len(embeddings)
        return embeddings


def get_next_messages(client, conversation_id, current_id,):
        return client.query(
                collection_name="message",
                filter=f"conversation_id != {int(conversation_id)} && id > {int(current_id)}",
                order='asc',
                limit=2,
                output_fields=["id", "user_id", "conversation_id", "prompt", "response", "timestamp"]
        )
         
def get_previous_messages(client, conversation_id, current_id,):
        return client.query(
                collection_name="message",
                filter=f"conversation_id != {int(conversation_id)} && id < {int(current_id)}",
                order='asc',
                limit=2,
                output_fields=["id", "user_id", "conversation_id", "prompt", "response", "timestamp"]
        )
         