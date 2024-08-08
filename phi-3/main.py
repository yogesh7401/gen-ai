import ollama
import chromadb

# Sample document
documents = [
  "Kimetsu no Yaiba, rgh. 'Blade of Demon Destruction' is a Japanese manga series written and illustrated by Koyoharu Gotouge.",
  "It was serialized in Shueisha's shōnen manga magazine Weekly Shōnen Jump from February 2016 to May 2020, with its chapters collected in 23 tankōbon volumes.",
  "It has been published in English by Viz Media and simultaneously on the Manga Plus platform by Shueisha.",
  "It follows teenage Tanjiro Kamado, who strives to become a Demon Slayer after his family was slaughtered and his younger sister, Nezuko, is turned into a demon.",
  "A 26-episode anime television series adaptation produced by Ufotable aired from April to September 2019, with a sequel film, Demon Slayer: Kimetsu no Yaiba - The Movie: Mugen Train, released in October 2020 and became the highest-grossing anime film and Japanese film of all time.",
  "An 18-episode second season of the anime series aired from October 2021 to February 2022 while a compilation film, Demon Slayer: Kimetsu no Yaiba - To the Swordsmith Village, was released in February 2023",
  "An 11-episode third season aired from April to June 2023 while another compilation film, Demon Slayer: Kimetsu no Yaiba - To the Hashira Training, was released in February 2024.",
  "A fourth season aired from May to June 2024.",
  "A film trilogy sequel was announced after the fourth season finale",
  "By February 2021, the manga had over 150 million copies in circulation, including digital versions, making it one of the best-selling manga series of all time.",
  "Also, it was the best-selling manga in 2019 and 2020.",
  "The manga has received critical acclaim for its art, storyline, action scenes and characters.",
  "The Demon Slayer: Kimetsu no Yaiba franchise is one of the highest-grossing media franchises of all time.",
  "A film trilogy adapting the 'Infinity Castle' story arc was announced after the fourth season finale in June 2024.",
]

client = chromadb.Client()
collection = client.create_collection(name="docs")

# Store embedded document in vector database
for index, doc in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=doc)
  embedding = response["embedding"]
  collection.add(
    ids=[str(index)],
    embeddings=[embedding],
    documents=[doc]
  )

# Query prompt
query_prompt = "What is best selling manga?"

# generate an embedding for the prompt and retrieve the most relevant document
response = ollama.embeddings(
  prompt=query_prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=5
)
data = results['documents'][0][0]
print(query_prompt)

# generate a response combining the prompt and data we retrieved
output = ollama.generate(
  model="phi3",
  prompt=f"Using this data as reference: {data}. Respond to this prompt: {query_prompt}"
)

print(output)
