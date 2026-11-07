import random

class NakliLLM:
    def __init__(self):
        print('LLM initialized...')
    
    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'Ai stands for artificial intelligence'
        ]
        
        return {'response' : random.choice(response_list)}
    
class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict=None, **kwargs):
        # Works whether you pass a dict {'length': '...'} or keywords length='...'
        values = input_dict if isinstance(input_dict, dict) else kwargs
        return self.template.format(**values)

class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.invoke(final_prompt)
        return result['response'] 



llm = NakliLLM()

# response = llm.invoke('Tell me something')
# print(response['response'])


template = NakliPromptTemplate(
    template='Write a {length} essay on the topic {topic}',
    input_variables=['length' , 'topic']
)

# prompt_output = template.format({'length':'short', 'topic':'AI'})
# print(prompt_output)


# prompt = template.format({'length':'short', 'topic':'AI'})
# response = llm.invoke(prompt)
# print(response['response'])



chain = NakliLLMChain(llm, template)
response = chain.run({'length' : 'short' , 'topic' : 'Cricket'})
print(response)