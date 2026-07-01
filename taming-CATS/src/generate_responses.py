import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env", override=True)
from classes.Metrics import Metrics


OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_TOKEN)


def generate_response(messages, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def simplify(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Simplify this text: {input_text}"}
    ]
    response = generate_response(messages)

    return response

def simplify_with_reasoning(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Briefly describe the needs of the target group, if such is specified. Using the instructions you were given, briefly explain how you would simplify the following source text and why: {input_text}. Think step by step, and explain the changes you would make to satisfy the instructions, including any calculations, if required. Do not yet generate the simplification! Be very brief."}
    ]
    reasoning = generate_response(messages)

    messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "user", "content": f"Now that you have developed a simplification strategy, generate the simplification. Only write the simplification. No other comments are allowed."})
    
    simplification = generate_response(messages)

    return reasoning, simplification

def simplify_with_consecutive_transformations(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"You will perform a step-by-step simplification of the following text: {input_text}."},
        {"role": "user", "content": f"Start by performing a syntactic simplification. Reduce sentences to minimal clauses. You can split sentences into several, if needed."}
    ]
    syntactic_symplification = generate_response(messages)

    messages.append({"role": "assistant", "content": syntactic_symplification})
    messages.append({"role": "user", "content": f"Proceed with a lexical simplification. Substitute domain-specific and difficult words with simple words, if possible."})

    lexical_simplification = generate_response(messages)

    messages.append({"role": "assistant", "content": lexical_simplification})
    messages.append({"role": "user", "content": f"Now, feel free to use paraphrase or explanation for terms and concepts you think require it."})

    paraphrase_or_explanation = generate_response(messages)

    messages.append({"role": "assistant", "content": paraphrase_or_explanation})
    messages.append({"role": "user", "content": f"Now, generate the final simplification based on your previous thoughts. Output only the simplification. No notes or comments are allowed."})

    simplification = generate_response(messages)

    return syntactic_symplification, lexical_simplification, paraphrase_or_explanation, simplification

def simplify_with_corrections(input_text, prompt):

    initial_simplification = simplify(input_text, prompt)

    messages = [
        {"role": "user", "content": f"A text simplification assistant was given the following task formulation: {prompt}."},
        {"role": "user", "content": f"This is the source text: {input_text}."},
        {"role": "user", "content": f"Given the task formulation and the source text, critically assess the simplification. Be brief (1-2 sentences)."}
    ]

    judgement = generate_response(messages)

    messages.append({"role": "assistant", "content": judgement})
    messages.append({"role": "user", "content": f"Take into account your critical assessment of the simplification. Now it is your turn to further improve upon the simplification to align it with the task, but make only necessary changes. Output only the simplification, no notes or comments are allowed."})

    final_simplification = generate_response(messages)

    return initial_simplification, judgement, final_simplification

def main():

    experiment_name = "basic" # reasoning, consecutive, basic, metacorrections
    dataset_name = "sample_sentence_newsela"
    # TODO setup iteration over our datasets
    input_text = "THE HAGUE, Netherlands \u2014 These days, anybody with a smartphone can snap a selfie in a split second. Back in the Dutch Golden Age, they were called self-portraits and were the preserve of highly trained artists who thought long and hard about every aspect of the painting.\n\nNow the Mauritshuis museum is staging an exhibition focusing solely on these 17th century self-portraits, highlighting the similarities and the differences between modern-day snapshots and historic works of art.\n\nThe museum's director, Emilie Gordenker, said recently there has never been such an exhibition of Golden Age Dutch self-portraits before and her museum was keen to tie the paintings to a modern-day phenomenon \u2014 the ubiquitous selfies captured with smartphone cameras and spread via social media.\n\nThe exhibition, opening Oct. 8 and running through Jan. 3, features 27 self-portraits by artists ranging from Rembrandt van Rijn, a master of the genre, to his student Carel Fabritius \u2014 best known for \"The Goldfinch,\" which hangs elsewhere in the Mauritshuis \u2014 and Judith Leyster, whose self-portrait is on loan from the National Gallery of Art in Washington, D.C.\n\nA less well-known artist, Huygh Pietersz Voskuyl, is the poster boy for the exhibition. His striking 1638 self-portrait features a classic selfie pose; staring over his right shoulder out of the frame. It does not take much imagination to picture him gazing into the lens of a smartphone rather than a mirror, which Golden Age artists used to capture their images for self-portraits. Giant mirrors are spread through the exhibition space, creating reflections within reflections of paintings that are themselves mirror images.\n\nWhile the similarities between selfies and self-portraits are obvious \u2014 the subject matter is the person creating the image \u2014 the differences are also apparent. A selfie is often shot speedily with little concern for composition, while these self-portraits are carefully conceived works of art. A video made for the exhibition highlights the thought that went into the paintings and what today's selfie makers can learn from it to improve their snapshots.\n\nAnd, yes, you are allowed to take selfies in the museum.\n\nThe Voskuyl is a good example of the richness that can be found in such an apparently simple picture.\n\n\"He brings out all these little details, like his beard or the little embroidery on his shirt, even a kind of fake wood-paneled wall behind him,\" Gordenker said. \"So he's thought very hard about the textures and the things that make him who he is. At the same time, you can see the skill with which he painted this and this will have definitely been a very good advertisement for what he could do.\"\n\nThat kind of attention to detail and quality made the self-portraits almost a Golden Age calling card \u2014 showcasing the artist and his or her talents to potential clients.\n\n\"A lot of artists in the 17th century painted self-portraits, not only as portraits of themselves but also as an example of the beautiful art that they could make,\" said the exhibition's curator Ariane van Suchtelen. \"For instance, Rembrandt was very famous for his very virtuoso sketchy way of painting. If you would buy a self-portrait by Rembrandt, you would not only have a portrait of this famous artist but also an example of what he could do, what he was famous for \u2014 his art.\""
    print("Input text:\n", input_text)

    # create output dir
    output_dir = f"./output/prompting/{experiment_name}/{MODEL}/{dataset_name}"
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with open("./data/prompts/prompts.json", "r") as file:
        prompts = json.load(file)
        with open("./data/prompts/justifications.json", "r") as file:
            justifications = json.load(file)

    # Iterate through prompt categories and variations
    for category, variations in prompts.items():
        for prompt_id, prompt in variations.items():

            if experiment_name == "reasoning":
                reasoning, simplified_text = simplify_with_reasoning(input_text, prompt)
                output_data = {
                    "reasoning": reasoning
                }
            elif experiment_name == "consecutive":
                syntactic, lexical, paraphrase, simplified_text = simplify_with_consecutive_transformations(input_text, prompt)
                output_data = {
                    "syntactic_simplification": syntactic,
                    "lexical_simplification": lexical,
                    "paraphrase_or_explanation": paraphrase
                }
            elif experiment_name == "metacorrections":
                initial_simplification, judgement, simplified_text = simplify_with_corrections(input_text, prompt)
                initial_simplification_metrics = Metrics(initial_simplification).compute_metrics()
                output_data = {
                    "initial_simplification": initial_simplification,
                    "initial_simplification_metrics": initial_simplification_metrics,
                    "metajudgement": judgement
                }

            else:
                simplified_text = simplify(input_text, prompt)
                output_data = {}

            source_metrics = Metrics(input_text).compute_metrics()
            target_metrics = Metrics(simplified_text).compute_metrics()
            output_data.update({
                "category": category,
                "prompt_used": prompt,
                "prompt_id": prompt_id,
                "source_text": input_text,
                "target_text": simplified_text,
                "source_metrics": source_metrics,
                "target_metrics": target_metrics
            })

            output_file = f"{output_dir}/{category}_{prompt_id}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as file:
                json.dump(output_data, file, indent=4)

            print(f"Output saved to {output_file}")


if __name__ == "__main__":
    main()