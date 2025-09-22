import re
import os
import argparse
import pandas as pd
from random import sample
from tqdm import tqdm
from datetime import datetime
from scripts.lvlm_chat import LVLMChat
from scripts.utils import get_conversations, load_dict_from_json
from scripts.prompt_templates import get_prompt_templates_from_setup_name


random_sequence_mapper = {
    10: "3, 7, 1, 5, 2, 8, 10, 4, 6, 9",
    13: "3, 7, 1, 12, 5, 2, 13, 8, 10, 4, 6, 9, 11",
}


SETUP_NAMES = [
    "one transcript at a time", "object summaries",
    "all transcripts", "object descriptions","plus feedback"
]


def get_args():
    parser = argparse.ArgumentParser(description="Run the observer task experiment.")
    # need something to check that the data_fp and image_dire correspond to each other
    parser.add_argument("--data_fp", type=str, default="data/baskets-matching-data.xlsx", help="Path to the data file")
    parser.add_argument("--image_dire", type=str, default="data/baskets-grid/", help="Directory containing images")
    parser.add_argument("--setup_name", type=str, default="one transcript at a time", choices=SETUP_NAMES, help="Setup name for the experiment")
    parser.add_argument("--rounds", type=str, default="1", help="Rounds of conversation to run. Default is 1,2,3,4")    
    parser.add_argument("--num_pairs", type=int, default=10, help="Number of pairs to run. Default is 10.")
    parser.add_argument("--num_experiments_per_experiment", type=int, default=1, help="Number of experiments per experiment")
    parser.add_argument("--models", type=str, default="openai/gpt-4o-mini-2024-07-18", help="Model name(s) to use")
    parser.add_argument("--max_img_dim", type=int, default=None, help="Max image dimension for the model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--output_fn", type=str, default=None, help="Output filename. If not provided, will be the timestamp.")
    parser.add_argument("--not_use_playbook", action="store_true", help="Whether to use the playbook. If not provided, will use the playbook.")
    parser.add_argument("--repeat_same_img", action="store_true", help="Whether to repeat the same image for each model run. Only implemented for setup_id=1 and setup_id=2.")
    
    return parser.parse_args()


def get_time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_prediction(response, num_objects_per_image):
    matches = re.findall(r'\d+(?:\s*,\s*\d+)+', response)
    
    if matches:
        for match in matches[::-1]:
            match = [int(x.strip()) for x in match.split(",")]
            if set(match) == set(range(1, num_objects_per_image)):
                return match
        
        return [int(x.strip()) for x in matches[-1].split(",")]
        
    return "CANNOT_PARSE"

    
def compute_accu(answer, pred):
    if pred == "CANNOT_PARSE":
        return 0
    else:
        return sum([a==p for a, p in zip(answer, pred)]) / len(answer)
    

def answer_transform(img_fp, answer, mapper):
    if img_fp in mapper:
        return [mapper[img_fp].index(x) + 1 for x in answer]
    else:
        raise ValueError(f"Image path {img_fp} not found in mapper.") 
    

def load_image_fps_and_mapper(image_dire):
    image_fps = [os.path.join(image_dire, f) for 
                 f in os.listdir(image_dire) if f.endswith('.png')]
    mapper = load_dict_from_json(os.path.join(image_dire, "mapper.json"))
    return image_fps, mapper


def extract_json_response(response, num_matches=None):    
    matches = re.findall(r'\d+(?:\s*,\s*\d+)+', response)[-num_matches:]

    if len(matches) == num_matches:
        out = dict()
        for i, match in enumerate(matches):
            out[f"Round {i+1}"] = [int(x.strip()) for x in match.split(",")]
        
        return out

    return "CANNOT_PARSE"


def compute_accu_for_json_response(answers, preds):
    out = [0] * len(answers)

    if preds == "CANNOT_PARSE":
        return out
    else:
        for i in range(len(preds)):
            try:
                pred = preds[f"Round {i+1}"]
                answer = answers[i]
                out[i] = sum([a==p for a, p in zip(answer, pred)]) / len(answer)
            except Exception as e:
                print(f"Error processing Round {i+1}: {e}")
                out[i] = "CANNOT_PARSE"
        
        return out


def main():
    args = get_args()
    print(args)

    obj1 = args.data_fp.split("/")[-1].split("-")[0]
    obj2 = args.image_dire.split("/")[-1].split("-")[0]
    assert obj1 == obj2, f"Data file {args.data_fp} and image directory {args.image_dire} do not match."

    df = pd.read_excel(args.data_fp)
    all_image_fps, mapper = load_image_fps_and_mapper(args.image_dire)
    num_objects_per_image = mapper["metadata"]["number_of_objects_per_image"]
    mapper = mapper["data"]

    playbook = load_dict_from_json(os.path.join(args.image_dire, "playbook.json"))

    system_prompt, prompt_tmp = get_prompt_templates_from_setup_name(args.setup_name)
    system_prompt = system_prompt.substitute(num_of_objects=num_objects_per_image, 
                                             example_sequence=random_sequence_mapper[num_objects_per_image])

    rounds = [int(t) for t in args.rounds.split(",")]
    pairs = df.columns.to_list()[3:][:args.num_pairs]
    models = args.models.split(",")

    if args.image_dire.endswith("/"):
        args.image_dire = args.image_dire[:-1]

    dire = args.image_dire.split("/")[-1]

    if args.output_fn is not None:
        res_fp = f"results/{dire}/setup{args.setup_name}/{args.output_fn}.csv"
    else:
        res_fp = f"results/{dire}/setup{args.setup_name}/{get_time_stamp()}.csv"
    os.makedirs(os.path.dirname(res_fp), exist_ok=True)

    if args.setup_name == "one transcript at a time":
        for i in range(len(rounds) - 1):
            assert rounds[i] + 1 == rounds[i + 1], \
                f"Under setup {args.setup_name}, rounds should be consecutive. Found {rounds[i]} and {rounds[i + 1]}."
         
        out = []
        cols = ["Run Number", "Round", "Pair", "Image FP", "Model", "Response", "Prediction", "Answer", "Accu"]

        for run_num in range(args.num_experiments_per_experiment):
            
            image_fps = sample(all_image_fps, len(rounds))
            
            for model in models:
                for pair in pairs:
                    chat = LVLMChat(model=model, system_prompt=system_prompt, 
                                    max_img_dim=args.max_img_dim, 
                                    temperature=args.temperature,)
                    
                    for i, round in enumerate(rounds):

                        if not args.not_use_playbook:
                            if args.repeat_same_img:
                                image_path = playbook[str(run_num)][str(rounds[0])]
                            else:
                                image_path = playbook[str(run_num)][str(round)]
                        else:
                            image_path = image_fps[i]
                        
                        transcript, answer = get_conversations(df, round, pair, return_entire_transcript=True)
                        answer = answer_transform(image_path, answer, mapper)

                        prompt = prompt_tmp.substitute(transcript=transcript, image_path=image_path)
                        response = chat.get_chat_completion(prompt)

                        try:
                            pred = extract_prediction(response, num_objects_per_image)
                            accu = compute_accu(answer, pred)
                            print(f"Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}, Accuracy {accu:.2f}")
                            out.append([run_num, round, pair, image_path, model, response, pred, answer, accu])
                        
                        except Exception as e:
                            print(f"Error processing Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}: {e}")
                            out.append([run_num, round, pair, image_path, model, response, "ERROR", answer, 0])
        
                    out_df = pd.DataFrame(out, columns=cols)
                    out_df.to_csv(res_fp, index=False)
        
        out_df = pd.DataFrame(out, columns=cols)
        out_df.to_csv(res_fp, index=False)

    elif args.setup_name == "object summaries":
        for i in range(len(rounds) - 1):
            assert rounds[i] + 1 == rounds[i + 1], \
                f"Under setup {args.setup_name}, rounds should be consecutive. Found {rounds[i]} and {rounds[i + 1]}."
         
        out = []
        cols = ["Run Number", "Round", "Pair", "Image FP", "Model", "Response", "Prediction", "Answer", "Accu"]

        for run_num in range(args.num_experiments_per_experiment):
            
            image_fps = sample(all_image_fps, len(rounds))
            
            for model in models:
                for pair in pairs:
                    chat = LVLMChat(model=model, system_prompt=system_prompt, 
                                    max_img_dim=args.max_img_dim, 
                                    temperature=args.temperature,)
                    
                    for i, round in enumerate(rounds):

                        if not args.not_use_playbook:
                            image_path = playbook[str(run_num)][str(round)]
                        else:
                            image_path = image_fps[i]
                        
                        summaries, answer = get_conversations(df, round, pair, return_entire_transcript=False)
                        summaries = ["### Summary for object " + str(i+1) + "\n" + s for i, s in enumerate(summaries)]
                        summaries = "\n\n".join(summaries)
                        answer = answer_transform(image_path, answer, mapper)
                        prompt = prompt_tmp.substitute(summaries=summaries, image_path=image_path)
                        response = chat.get_chat_completion(prompt)

                        try:
                            pred = extract_prediction(response, num_objects_per_image)
                            accu = compute_accu(answer, pred)
                            print(f"Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}, Accuracy {accu:.2f}")
                            out.append([run_num, round, pair, image_path, model, response, pred, answer, accu])
                        
                        except Exception as e:
                            print(f"Error processing Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}: {e}")
                            out.append([run_num, round, pair, image_path, model, response, "ERROR", answer, 0])
        
                    out_df = pd.DataFrame(out, columns=cols)
                    out_df.to_csv(res_fp, index=False)
        
        out_df = pd.DataFrame(out, columns=cols)
        out_df.to_csv(res_fp, index=False)
    
    elif args.setup_name == "all transcripts":
        for i in range(len(rounds) - 1):
            assert rounds[i] + 1 == rounds[i + 1], \
                f"Under setup {args.setup_name}, rounds should be consecutive. Found {rounds[i]} and {rounds[i + 1]}."
            
        out = []
        cols = ["Run Number", "Rounds", "Pair", "Image FPs", "Model", "Response", "Prediction", "Answer", "Accu"]

        for run_num in range(args.num_experiments_per_experiment):
            image_fps = sample(all_image_fps, len(rounds))
            
            for model in models:
                for pair in pairs:
                    chat = LVLMChat(model=model, 
                                    system_prompt=system_prompt, 
                                    max_img_dim=args.max_img_dim, 
                                    temperature=args.temperature,)
                    prompts, answers = [], []

                    for i, round in enumerate(rounds):

                        if not args.not_use_playbook:
                            image_path = playbook[str(run_num)][str(round)]
                            image_fps[i] = image_path
                        else:
                            image_path = image_fps[i]
                        
                        transcript, answer = get_conversations(df, round, pair, return_entire_transcript=True)
                        answer = answer_transform(image_path, answer, mapper)
                        prompt = prompt_tmp.substitute(transcript=transcript, image_path=image_path, ix=i+1)
                        prompts.append(prompt)
                        answers.append(answer)
                    
                    prompt = f"\n\n{'*'*50}\n\n".join(prompts)
                    response = chat.get_chat_completion(prompt)
                    try:
                        preds = extract_json_response(response, num_matches=len(rounds))
                        accus = compute_accu_for_json_response(answers, preds)
                        print(f"Run Number {run_num}, Image Paths {image_fps}, Rounds {rounds}, Pair {pair}, Model {model}, Accuracy {accus}")
                        out.append([run_num, rounds, pair, image_fps, model, response, preds, answers, accus])
                    
                    except Exception as e:
                        print(f"Error processing Run Number {run_num}, Image Paths {image_fps}, Rounds {rounds}, Pair {pair}, Model {model}: {e}")
                        out.append([run_num, rounds, pair, image_fps, model, response, "ERROR", answers, 0])
        
                out_df = pd.DataFrame(out, columns=cols)
                out_df.to_csv(res_fp, index=False)

        out_df = pd.DataFrame(out, columns=cols)
        out_df.to_csv(res_fp, index=False)
        print(f"Results saved to {res_fp}")
    
    elif args.setup_name == "plus feedback":
        for i in range(len(rounds) - 1):
            assert rounds[i] + 1 == rounds[i + 1], \
                f"Under setup {args.setup_name}, rounds should be consecutive. Found {rounds[i]} and {rounds[i + 1]}."
         
        out = []
        cols = ["Run Number", "Round", "Pair", "Image FP", "Model", "Response", "Prediction", "Answer", "Accu"]

        for run_num in range(args.num_experiments_per_experiment):
            
            image_fps = sample(all_image_fps, len(rounds))
            
            for model in models:
                for pair in pairs:
                    chat = LVLMChat(model=model, system_prompt=system_prompt, 
                                    max_img_dim=args.max_img_dim, 
                                    temperature=args.temperature,)
                    
                    for i, round in enumerate(rounds):

                        if not args.not_use_playbook:
                            if args.repeat_same_img:
                                image_path = playbook[str(run_num)][str(rounds[0])]
                            else:
                                image_path = playbook[str(run_num)][str(round)]
                        else:
                            image_path = image_fps[i]
                        
                        transcript, answer = get_conversations(df, round, pair, return_entire_transcript=True)
                        answer = answer_transform(image_path, answer, mapper)

                        prompt = prompt_tmp.substitute(transcript=transcript, image_path=image_path)
                        response = chat.get_chat_completion(prompt)

                        try:
                            pred = extract_prediction(response, num_objects_per_image)
                            accu = compute_accu(answer, pred)
                            print(f"Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}, Accuracy {accu:.2f}")
                            out.append([run_num, round, pair, image_path, model, response, pred, answer, accu])
                        
                        except Exception as e:
                            print(f"Error processing Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}: {e}")
                            out.append([run_num, round, pair, image_path, model, response, "ERROR", answer, 0])
                        
                        answer_prompt = f"Here is correct sequence of picture indices as described by the Director: {answer}. Reflect on your previous answer if it was wrong. We will proceed after your reflection."
                        response = chat.get_chat_completion(answer_prompt)
                        out.append([run_num, round, pair, image_path, model, response, "ResponseToTheCorrectAnswer", answer, 0])
        
                    out_df = pd.DataFrame(out, columns=cols)
                    out_df.to_csv(res_fp, index=False)
        
        out_df = pd.DataFrame(out, columns=cols)
        out_df.to_csv(res_fp, index=False)


    elif args.setup_name == "object descriptions":
        assert args.num_experiments_per_experiment <= len(all_image_fps), \
            f"Under setup {args.setup_name}, number of experiments per experiment {args.num_experiments_per_experiment} "\
                f"cannot be greater than number of images {len(all_image_fps)}."
        
        out = []
        cols = ["Run Number", "Round", "Pair", "Image FP", "Model", "Conversation", "Response", "Prediction", "Answer", "Accu"]

        image_fps = sample(all_image_fps, args.num_experiments_per_experiment)

        for run_num, image_path in enumerate(image_fps):
            for model in models:
                for round in rounds:

                    if not args.not_use_playbook:
                        image_path = playbook[str(run_num)][str(round)]
                    
                    for pair in pairs:
                        conversations, answer = get_conversations(df, round, pair, return_entire_transcript=False)
                        conversations = conversations + ["The conversation is over. Please give your final answer."]
                        answer = answer_transform(image_path, answer, mapper)

                        chat = LVLMChat(model=model,
                                        system_prompt=system_prompt, 
                                        max_img_dim=args.max_img_dim,
                                        temperature=args.temperature)
                        
                        prompt = prompt_tmp.substitute(conversation=conversations[0], image_path=image_path)
                        response = chat.get_chat_completion(prompt)
                        out.append([run_num, round, pair, image_path, model, conversations[0], response, '-', '-', '-'])

                        for ix, conversation in tqdm(enumerate(conversations[1:])):
                            response = chat.get_chat_completion(conversation)
                            out.append([run_num, round, pair, image_path, model, conversation, response, '-', '-', '-'])
                        
                        # final response
                        try:
                            pred = extract_prediction(response, num_objects_per_image)
                            accu = compute_accu(answer, pred)
                            print(f"Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}, Accuracy {accu:.2f}")
                            out.append([run_num, round, pair, image_path, model, conversation, response, pred, answer, accu])
                        
                        except Exception as e:
                            print(f"Error processing Run Number {run_num}, Image Path {image_path}, Round {round}, Pair {pair}, Model {model}: {e}")
                            out.append([run_num, round, pair, image_path, model, conversation, response, "ERROR", answer, 0])
                
                    out_df = pd.DataFrame(out, columns=cols)
                    out_df.to_csv(res_fp, index=False)
        
        out_df = pd.DataFrame(out, columns=cols)
        out_df.to_csv(res_fp, index=False)
        print(f"Results saved to {res_fp}")


if __name__ == "__main__":
    main()