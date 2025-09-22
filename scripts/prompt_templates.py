from string import Template


PLUS_FORMAL_PROMPT_TEMPLATE = '''
You are given an excerpt from a transcribed, spontaneous conversation between two individuals. \
Your task is to revise the excerpt to produce a clear, polished version of the dialogue that reads like formal written text. \
Transform any standalone words or phrases into complete, grammatically correct sentences where appropriate. Do not add any \
additional information or context or change the meaning of the text. Do not output anything other than the revised excerpt.

Here is the excerpt:
${excerpt}

Revised Excerpt:
'''.strip()
PLUS_FORMAL_PROMPT_TEMPLATE = Template(PLUS_FORMAL_PROMPT_TEMPLATE)


MINUS_INTERACTION_PROMPT_TEMPLATE = '''
You are given an excerpt from a transcribed, spontaneous conversation between two individuals. \
Your task is to extract and concisely summarize all descriptions used to characterize a specific object mentioned in the excerpt. \
You must follow the instructions below:

1. Preserve all relevant descriptive details.
2. Do not alter the meaning, add context, or introduce new information.
3. Your response must only include the final summary—do not include the original excerpt or any explanatory text.

Excerpt:

${excerpt}

Summary of Object Descriptions:
'''.strip()
MINUS_INTERACTION_PROMPT_TEMPLATE = Template(MINUS_INTERACTION_PROMPT_TEMPLATE)


def get_one_transcript_at_a_time_prompt_templates():
    system_prompt = '''
You are an overhearer of a conversation between two participants engaged in a collaborative object-matching task for one or multiple rounds. \
Each participant is in a separate room and has a duplicate set of pictures arranged in different random orders. \
They cannot see each other’s sets and communicate solely via an audio link. \
During the task, one participant acts as the Director (D) and the other as the Matcher (M). \
The Director describes the pictures one at a time, and the Matcher selects the corresponding picture from their own set. \
Please note that it is the same two participants playing the same roles for all the rounds if there are multiple rounds. 

As the overhearer and for each round, you are provided with: 

- The full transcript of their conversation for that round.
- An image showing all pictures used in the task, randomly arranged and labeled with indices from 1 to $num_of_objects. The image for each round may be different.

Your goal is to determine the correct sequence of picture indices as described by the Director during each round. To do this:

1. Carefully analyze the transcript to understand which pictures the Director refers to, in the order they were described.
2. Use the image to match each described picture to its corresponding index.
3. Think step by step and revise your reasoning and answers as needed. \
However, you may not ask questions or make assumptions beyond the given materials.

When you reach your conclusion, output your response in the following format:

Final Answer: [$num_of_objects picture indices in correct order, separated by commas]

Example: $example_sequence
'''.strip()

    prompt_tmp = '''
The transcript of the current conversation is as follows:

$transcript

The image for the current round showing the pictures is as follows:

<$image_path>
'''.strip()
    
    return Template(system_prompt), Template(prompt_tmp)



def get_object_summaries_prompt_templates():
    system_prompt = '''
You are an overhearer of a conversation between two participants engaged in a collaborative object-matching task for one or multiple rounds. \
Each participant is in a separate room and has a duplicate set of pictures arranged in different random orders. \
They cannot see each other’s sets and communicate solely via an audio link. \
During the task, one participant acts as the Director (D) and the other as the Matcher (M). \
The Director describes the pictures one at a time, and the Matcher selects the corresponding picture from their own set. \
Please note that it is the same two participants playing the same roles for all the rounds if there are multiple rounds. 

As the overhearer and for each round, you are provided with: 

- 10 object summaries based on the Director's description of the target pictures for that round. 
- An image showing all pictures used in the task, randomly arranged and labeled with indices from 1 to $num_of_objects. The image for each round may be different.

Your goal is to determine the correct sequence of picture indices as described by the Director during each round. To do this:

1. Carefully analyze the transcript to understand which pictures the Director refers to, in the order they were described.
2. Use the image to match each described picture to its corresponding index.
3. Think step by step and revise your reasoning and answers as needed. \
However, you may not ask questions or make assumptions beyond the given materials.

When you reach your conclusion, output your response in the following format:

Final Answer: [$num_of_objects picture indices in correct order, separated by commas]

Example: $example_sequence
'''.strip()

    prompt_tmp = '''
The 10 object summaries based on the Director's description are as follows:

$summaries

The image for the current round showing the pictures is as follows:

<$image_path>
'''.strip()
    
    return Template(system_prompt), Template(prompt_tmp)



def get_object_descriptions_prompt_templates():
    system_prompt = '''
You are an overhearer of an ongoing conversation between two participants engaged in a collaborative object-matching task. \
Each participant is in a separate room and has a duplicate set of pictures arranged in different random orders. \
They cannot see each other’s sets and they can communicate solely via an audio link. \
During the task, one participant acts as the Director (D) and the other as the Matcher (M). \
The Director describes the pictures one at a time, and the Matcher selects the corresponding picture from their own set. 

As the overhearer and for each target picture, you are provided with: 

- An image showing all pictures used in the task, randomly arranged and labeled with indices from 1 to $num_of_objects.
- Conversation between the Director (D) and Matcher (M) where the Matcher indicates that they have selected a target picture. 

Your goal is to determine the correct sequence of picture indices as described by the Director during the task. To do this:

1. Carefully analyze each conversation to understand which pictures the Director refers to, in the order they were described.
2. Use the image to match each described picture to its corresponding index.
3. Think step by step and revise your reasoning and answers as needed. \
However, you may not ask questions or make assumptions beyond the given materials.

You should produce a target picture index for each conversation presented to you as your current best guess. \
Once all the 10 pictures have been selected by the Matcher, you should reach a final conclusion and output your response in the following format:

Final Answer: [$num_of_objects picture indices in correct order, separated by commas]

Example: $example_sequence
'''.strip()

    prompt_tmp = '''
The image showing the pictures is as follows:

<$image_path>

The conversation between the Director (D) and Matcher (M) for the first target picture is as follows:

$conversation
'''.strip()

    return Template(system_prompt), Template(prompt_tmp)



def get_all_transcripts_prompt_templates():
    system_prompt = '''
You are an overhearer of a conversation between two participants engaged in a collaborative object-matching task for multiple rounds. \
Each participant is in a separate room and has a duplicate set of pictures arranged in different random orders. \
They cannot see each other’s sets and communicate solely via an audio link. \
During the task, one participant acts as the Director (D) and the other as the Matcher (M). \
The Director describes the pictures one at a time, and the Matcher selects the corresponding picture from their own set. \
Please note that it is the same two participants playing the same roles for all the rounds. 

As the overhearer, you are provided with: 

- The full transcripts of their conversation for each round.
- Images showing all pictures used in the task for each round, randomly arranged and labeled with indices from 1 to $num_of_objects. 

Your goal is to determine the correct sequence of picture indices as described by the Director for each round. To do this:

1. Carefully analyze the transcript to understand which pictures the Director refers to, in the order they were described.
2. Use the image to match each described picture to its corresponding index.
3. Think step by step and revise your reasoning and answers as needed. \
However, you may not ask questions or make assumptions beyond the given materials.

When you reach your conclusion, output your response in the following JSON format for each round:

Final Answer: {"Round i": [$num_of_objects picture indices in correct order, separated by commas]}

Example: {"Round 1": [$example_sequence], ...}
'''.strip()

    prompt_tmp = '''
The transcript of the conversation during round#$ix is as follows:

$transcript

The image  for the round#$ix showing the pictures is as follows:

<$image_path>
'''.strip()

    return Template(system_prompt), Template(prompt_tmp)


def get_plus_feedback_prompt_templates():
    system_prompt = '''
You are an overhearer of a conversation between two participants engaged in a collaborative object-matching task for one or multiple rounds. \
Each participant is in a separate room and has a duplicate set of pictures arranged in different random orders. \
They cannot see each other’s sets and communicate solely via an audio link. \
During the task, one participant acts as the Director (D) and the other as the Matcher (M). \
The Director describes the pictures one at a time, and the Matcher selects the corresponding picture from their own set. \
Please note that it is the same two participants playing the same roles for all the rounds if there are multiple rounds. 

As the overhearer and for each round, you are provided with: 

- The full transcript of their conversation for that round.
- An image showing all pictures used in the task, randomly arranged and labeled with indices from 1 to $num_of_objects. The image for each round may be different.
- After you provide your answer and before the next round starts, you will be provided with the correct sequence of picture indices as described by the Director for that round. 

Your goal is to determine the correct sequence of picture indices as described by the Director during each round. To do this:

1. Carefully analyze the transcript to understand which pictures the Director refers to, in the order they were described.
2. Use the image to match each described picture to its corresponding index.
3. Think step by step and revise your reasoning and answers as needed. \
However, you may not ask questions or make assumptions beyond the given materials.

When you reach your conclusion, output your response in the following format:

Final Answer: [$num_of_objects picture indices in correct order, separated by commas]

Example: $example_sequence
'''.strip()

    prompt_tmp = '''
The transcript of the current conversation is as follows:

$transcript

The image for the current round showing the pictures is as follows:

<$image_path>
'''.strip()
    
    return Template(system_prompt), Template(prompt_tmp)



def get_prompt_templates_from_setup_name(setup_name):
    if setup_name == "one transcript at a time":
        return get_one_transcript_at_a_time_prompt_templates()
    elif setup_name == "object summaries":
        return get_object_summaries_prompt_templates()
    elif setup_name == "all transcripts":
        return get_all_transcripts_prompt_templates()
    elif setup_name == "object descriptions":
        return get_object_descriptions_prompt_templates()
    elif setup_name == "plus feedback":
        return get_plus_feedback_prompt_templates()
    else:
        raise ValueError(f"Not Implemented for setup_name {setup_name}.")