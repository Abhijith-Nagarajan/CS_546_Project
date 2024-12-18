import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load triplets from file
def load_triplets(file_path):
    triplets = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip("()\n").split(", ")
                if len(parts) == 3:
                    triplets.append((parts[0].strip("'"), parts[1].strip("'"), parts[2].strip("'")))
    return triplets

# Build the knowledge graph
def BuildKG(triplets):
    G = nx.DiGraph()
    RELdir = {}
    for head, relation, tail in triplets:
        G.add_edge(head, tail, label=relation)
        RELdir[(head, tail)] = relation
    return G, RELdir

# Build relation embeddings
def build_relation_embeddings(triplets, model):
    """Create embeddings for triplet relations using Sentence Transformers."""
    relations = [" ".join([head, relation, tail]) for head, relation, tail in triplets]
    relation_embeddings = model.encode(relations, convert_to_tensor=True)
    return relations, relation_embeddings

# Retrieve top-k relations
def retrieve_using_embeddings(query, relations, relation_embeddings, model, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity_scores = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1),
                                          relation_embeddings.cpu().numpy())[0]
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    return [(relations[i], similarity_scores[i]) for i in top_indices]

# Query Expansion using GPT-Neo
def expand_query_with_gpt(query):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    prompt = f"Expand the query: '{query}' into more specific and related questions."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
    expanded_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return expanded_query.split(". ")

# Ground Truth Answers
def hardcoded_ground_truth():
    return [
        "p53 is positively correlated with MDM2 and cancer.",
        "RNA polymerase II is associated with 5S ribosomal RNA.",
        "GATA-1 is associated with hematopoietic factor and has a subclass relationship.",
        "The spinal cord is connected to cerebellum, thalamus, and pineal region."
    ]

# Evaluate cosine similarity
def evaluate_similarity_with_ground_truth(model, generated_responses, ground_truth):
    scores = []
    for gen, truth in zip(generated_responses, ground_truth):
        gen_embedding = model.encode(gen, convert_to_tensor=True).cpu().numpy()
        truth_embedding = model.encode(truth, convert_to_tensor=True).cpu().numpy()
        similarity = cosine_similarity(gen_embedding.reshape(1, -1), truth_embedding.reshape(1, -1))
        scores.append(similarity[0][0])
    return scores

# Main Execution
if __name__ == "__main__":
    # Load triplets
    file_path = '/Users/sanyamjain/Desktop/CS546_project/aggregated_triplets.txt'
    triplets = load_triplets(file_path)
    print(f"Loaded {len(triplets)} triplets from file.")

    # Build knowledge graph
    G, RELdir = BuildKG(triplets)
    print("Knowledge graph built successfully.")

    # Initialize Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Build relation embeddings
    relations, relation_embeddings = build_relation_embeddings(triplets, model)
    print("Relation embeddings built successfully.")

    # Queries
    queries = [
        "Which entity is positively correlated with p53?",
        "What association exists between RNA polymerase II and 5S ribosomal RNA?",
        "What is the relationship between GATA-1 and hematopoietic factor?",
        "Which entity is connected to spinal cord?"
    ]

    # Hardcoded ground truths
    ground_truth = hardcoded_ground_truth()

    # Retrieve responses using semantic search
    generated_responses = []
    for query in queries:
        print(f"\nOriginal Query: {query}")
        expanded_queries = expand_query_with_gpt(query)
        print(f"Expanded Queries: {expanded_queries}")

        # Retrieve best matches
        best_response = None
        best_score = -1
        for expanded_query in expanded_queries:
            top_matches = retrieve_using_embeddings(expanded_query, relations, relation_embeddings, model)
            if top_matches and top_matches[0][1] > best_score:
                best_response, best_score = top_matches[0]

        generated_responses.append(best_response if best_response else "No information found.")

    # Evaluate similarity with ground truths
    similarity_scores = evaluate_similarity_with_ground_truth(model, generated_responses, ground_truth)

    # Print Results
    print("\nFinal Results:")
    for i, (query, generated, truth, score) in enumerate(zip(queries, generated_responses, ground_truth, similarity_scores)):
        print(f"\nQuery {i+1}: {query}")
        print(f"Generated Response: {generated}")
        print(f"Ground Truth: {truth}")
        print(f"Cosine Similarity Score: {score:.4f}")
