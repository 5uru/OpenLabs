from typing import cast

import huggingface_hub.constants
import numpy as np
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from uform import Modality, get_model
from usearch.index import Index

if not hasattr(huggingface_hub.constants, "HF_HUB_ENABLE_HF_TRANSFER"):
    setattr(huggingface_hub.constants, "HF_HUB_ENABLE_HF_TRANSFER", False)

processors, models = get_model("unum-cloud/uform3-image-text-english-small")
model_text = models[Modality.TEXT_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]

Base = declarative_base()


class MyTable(Base):
    __tablename__ = "my_table"
    id = Column(Integer, primary_key=True)
    text = Column(String)


# Create engine and tables
engine = create_engine("sqlite:///mydatabase.db")
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()


index = Index(ndim=256)


class VectorDB:
    def __init__(self):
        self.session = session

    def add_entry(self, text, text_embedding):
        new_entry = MyTable(text=text)
        self.session.add(new_entry)
        self.session.commit()

        text_data = processor_text(text_embedding)
        text_embedding = model_text.encode(text_data, return_features=False)
        index.add(cast(int, new_entry.id), text_embedding.flatten(), copy=True)

    def get_entry(self, query, top_k=10):
        tokens = processor_text(query)
        vector = model_text.encode(tokens, return_features=False)
        matches = index.search(vector.flatten(), top_k)
        matches_ids = [int(id) for id in matches.keys]
        return [self.session.query(MyTable).get(id).text for id in matches_ids]
