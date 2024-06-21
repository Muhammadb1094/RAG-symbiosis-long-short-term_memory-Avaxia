from chat.models import Message
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from pymilvus import model
from .utils import get_next_messages, get_previous_messages



class ParserClass(BaseModel):
    setup: str = Field(description="User query to LLM")
    punchline: str = Field(description="answer to the question user asked.")
    knows_answer: bool = Field(
        description="Whether or not the LLM knows the answer to the question. Or it says I dont know."
    )


def check_gpt_knows_answer_yes_no(query, answer):
    model = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    parser = PydanticOutputParser(pydantic_object=ParserClass)

    prompt = PromptTemplate(
        template="The query user asked:\n{format_instructions}\n{query}\n The answer user got from AI: \n {answer}",
        input_variables=["query", "answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | model | parser
    result = chain.invoke({"query": query, "answer": answer})

    # print(result)
    return result.knows_answer
    

def short_term_memory_llm(query, conver_id):
    print("::: Short Term Memory calling :::")

    memory = ConversationBufferMemory()
    try:
        if not conver_id == "" or not conver_id is None:
            memorybuffer = Message.objects.filter(
                conversation_id=conver_id).order_by('-created_at')

            for item in memorybuffer:
                user_message = f"{item.query}. At datetime {item.created_at}"
                ai_message = f"{item.response}. At datetime {item.updated_at}"
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(ai_message)
            memory.load_memory_variables({})
    except Exception as e:
        print(e)

    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory,
    )
    output = conversation.predict(input=query)
    return output


def long_term_memory_llm(client, user, prompt, response, conversation_id):
    print("::: Long Term Memory calling :::")
    
    embedding_fn = model.DefaultEmbeddingFunction()

    memorybuffer = client.search(
        collection_name="message",
            filter=f"conversation_id != {int(conversation_id)} && user_id == {int(user.id)}",
            data=embedding_fn.encode_queries([prompt]),
            output_fields=["id", "user_id", "conversation_id",
                                "prompt", "response", "timestamp"],
            limit=1
        )
    print("::: Memory Buffer :::", memorybuffer)
    # try:
    memory = ConversationBufferMemory()
    # Load previous 2 messages from the relevant chat
    previous_chat = get_previous_messages(client, conversation_id, memorybuffer[0][0]['entity']['id'])
    for item in previous_chat:
        user_message = f"{item['prompt']}. At datetime {item['timestamp']}"
        ai_message = f"{item['response']}. At datetime {item['timestamp']}"
        
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(ai_message)
    print("::: Previous Chat :::", previous_chat)
    # Load Relevant message
    for item in memorybuffer[0]:
        user_message = f"{item['entity']['prompt']}. At datetime {item['entity']['timestamp']}"
        ai_message = f"{item['entity']['response']}. At datetime {item['entity']['timestamp']}"

        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(ai_message)
    
    # Lead the Next Relevant Messages
    next_chat = get_next_messages(client, conversation_id, memorybuffer[0][0]['entity']['id'])
    for item in next_chat:
        user_message = f"{item['prompt']}. At datetime {item['timestamp']}"
        ai_message = f"{item['response']}. At datetime {item['timestamp']}"
        
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(ai_message)
    
    print("::: Next Chat :::", next_chat)

    memory.load_memory_variables({})
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory,
    )
    return conversation.predict(input=prompt)

    # except Exception as e:
    #     print(e)
    #     return response


