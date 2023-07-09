from langchain.llms import OpenAI
import openai

def openai_request_original(input):

    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )

def openai_request_lang(input):

    llm = OpenAI(temperature=0.9)
    print(llm(text))

def generate_gpt3_response(user_text, print_output=True):
    """
    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    completions = openai.Completion.create(
        engine='text-davinci-003',  # Determines the quality, speed, and cost.
        temperature=0.5,            # Level of creativity in the response
        prompt=user_text,           # What the user typed in
        max_tokens=100,             # Maximum tokens in the prompt AND response
        n=1,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )

    # Displaying the output can be helpful if things go wrong
    if print_output:
        print(completions)

    # Return the first choice's text
    return completions.choices[0].text

if __name__ == '__main__':
    text = "What would be a good company name for a company that makes colorful socks?"
    text2 = "Three Rings for the elven lords, "
    prompt = "Write me a python function for scraping a dynamic website and use it as input for an openai llm"
    #openai_request_lang(text)
    generate_gpt3_response(prompt)


