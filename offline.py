from services.indexing import Indexing

Indexing.generate_index("Datasets/wikIR1k/documents.csv", "wikIR1k", version="05")
Indexing.generate_index("Datasets/antique/antique-collection.txt", "antique", version="05")