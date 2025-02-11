import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)

# Prompt
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# # Generate the output
# output = pipe(messages)
# print(output[0]["generated_text"])

# # Apply prompt template
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
# print(prompt)

# Prompt components
persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries.\n"
instruction = "Summarize the key findings of the paper provided.\n"
context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper.\n"
data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results.\n"
audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models.\n"
tone = "The tone should be professional and clear.\n"
text = "Papadopoulos worked an unpaid intern at the Hudson Institute from 2011 to 2015 specializing in the eastern Mediterranean[16] and later worked as a contract research assistant to a senior fellow at the institute.[19] Richard Weitz, a Wikistrat expert,[20] managed George Papadopoulos while he was at the Hudson Institute.[21] According to CNN, Papadopoulos described himself as an oil, gas, and policy consultant on his LinkedIn profile.[22] In 2010, following the rupture in relations between Turkey and Israel due to the Mavi Marmara incident, Papadopoulos became involved in eastern Mediterranean energy development projects and policy focusing upon the relations of Israel, Cyprus, Greece, also known as the Energy Triangle.[23] In 2014, Papadopoulos authored op-ed pieces in several Israeli publications. In one, published in the Arutz Sheva, Papadopoulos argued that the U.S. should focus on its stalwart allies Israel, Greece, and Cyprus to contain the newly emergent Russian fleet; in another, published in Ha'aretz, he contended that Israel should exploit its natural gas resources in partnership with Cyprus and Greece rather than Turkey.[24] On 19 October 2015, Russian President Vladimir Putin and Israeli Prime Minister Benjamin Netanyahu agreed to allow major concessions for Gazprom to develop the Leviathan gas field; Putin told Netanyahu, We will make sure there will be no provocation against the gas fields by Hezbollah or Hamas. Nobody messes with us.[25] Investigative reporting conducted by Ha'aretz in 2017 showed that Papadopoulos co-authored an expert opinion, on behalf of the Hudson Institute that was delivered to the Israeli Energy Ministry on June 20, 2015, about a proposed plan to develop the Leviathan offshore gas fields in Israel's territorial waters. Money was donated to Hudson by the CEO of Noble Energy and other staffers of the company. Houston-based Noble Energy is heavily invested in Israeli gas with the Israeli energy group Delek Drilling.[26] Noble Energy was initially given permission on December 17, 2015, to develop the Leviathan gas field worth up to $120 billion (~$151 billion in 2023).[27] In September 2015, Papadopoulos left the Hudson Institute and joined Energy Stream, a London energy consultancy, as an oil and gas consultant for four months before joining Ben Carson's presidential campaign.[28] Beginning in December 2015 and while living in London, Papadopoulos served on the National Security and Foreign Policy Advisory Committee for Ben Carson's campaign for the 2016 Republican presidential nomination.[29] In early February 2016, he began work as a director at the London Centre of International Law Practice[30] but left the Carson campaign in mid-February 2016 and moved from London to Chicago in March 2016.[3][16] Former Trump campaign adviser Michael Caputo described Papadopoulos role in the Trumps 2016 election campaign as a volunteer coffee boy.[31] In 2019, Papadopoulos announced he had joined the board of advisors for the medical cannabis company C3, which manufactures the marijuana pill Idrasil.[32][33] In September 2024, reports emerged that George Papadopoulos had joined the editorial board of the website Intelligencer, alongside contributors associated with Kremlin-affiliated media.[34] The company associated with Intelligencer lists a Los Angeles business "
data = f"Text to summarize: {text}"

# The full prompt - remove and add pieces to view its impact on the generated output
query = persona + instruction + context + data_format + audience + tone + data

output = pipe(query)
print(output[0]["generated_text"])