
import os 
from secret_key import openapi_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain



os.environ["OPEN_API_KEY"] = openapi_key


llm = OpenAI(temperature = 0.7)

def generate_restaurant_name_and_items(cuisine):

    # Chain1:Restaurant Name
    llm = OpenAI(temperature= 0.6)
    PromptTemplate_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food.Suggest a fancy name for this"
    )
    name_chain = LLMChain(llm = llm,prompt = PromptTemplate_name,output_key = "restaurant_name")

    # Chain 2:Menu Items
    PromptTemplate_items = PromptTemplate(

        input_variables=["restaurant_name"],
        template="Suggest some menu items for {restaurant_name}."
    )

    food_items_chain = LLMChain(llm = llm,prompt = PromptTemplate_items,output_key = "menu_items")

    #SequentialChain
    chains = SequentialChain(
        chains = [name_chain,food_items_chain],
        input_variables = ["cuisine"],
        output_variables = ["restaurant_name","menu_items"]

    )

    response = chains({"cuisine":cuisine})
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))






