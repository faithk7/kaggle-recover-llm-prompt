from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

"""
For each row in the submission and corresponding ground truth, sentence-t5-base is used to calculate corresponding embedding vectors.
The score for each predicted / expected pair is calculated using the Sharpened Cosine Similarity, using an exponent of 3.
The SCS is used to attenuate the generous score given by embedding vectors for incorrect answers. Do not leave any rewrite_prompt blank as null answers will throw an error.
"""
model_name = "sentence-transformers/sentence-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
model = SentenceTransformer(model_name)


def calculate_score(generated_sentence, ground_truth_sentence):
    submission_embedding = get_embedding(generated_sentence)
    ground_truth_embedding = get_embedding(ground_truth_sentence)
    scs = sharpened_cosine_similarity(submission_embedding, ground_truth_embedding)
    return scs


# def get_embedding(sentence):
#     inputs = tokenizer(
#         sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
#     )
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings[0].numpy()


def get_embedding(sentence):
    embedding = model.encode(sentence)
    return embedding


def sharpened_cosine_similarity(embedding1, embedding2, exponent=3):
    cosine_sim = 1 - cosine(embedding1, embedding2)
    scs = cosine_sim**exponent
    return scs


if __name__ == "__main__":
    generated_sentence = "What is the capital of France?"
    # generated_sentence = "Paris is the capital of France."
    ground_truth_sentence = "Paris is the capital of France."

    score = calculate_score(generated_sentence, ground_truth_sentence)
    print(f"Score: {score:.4f}")
