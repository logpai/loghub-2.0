
import os
from typing import List

# from spacy.lang.id import Indonesian
import spacy
from gensim.models import Word2Vec, KeyedVectors
from spacy.language import Language


class NewsPreprocessor(object):

    def __init__(self):
        self.sentences: List[List[str]] = []
        self.nlp: Language =spacy.load("en_core_web_sm")
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def read_news(self, root_folder: str) -> List[List[str]]:
        child_folders = os.listdir(root_folder)
        num_folders = len(child_folders)
        proc_sentences = []
        for i in range(num_folders):
            print(f"Processing folder: {i+1}/{num_folders}")
            folder = child_folders[i]
            txt_files = os.listdir(os.path.join(root_folder, folder))
            for txt_file in txt_files:
                with open(os.path.join(root_folder, folder, txt_file), "r") as f:
                    article = f.readlines()
                for p in article:
                    p_clean = p.replace("\n", "")
                    if len(p_clean):
                        doc = self.nlp(p_clean)
                        for sent in doc.sents:
                            if len(sent) > 2:
                                joined_tokens = " ".join(
                                    [str(token).strip() for token in sent if len(str(token).strip()) > 0]
                                ).strip()
                                proc_sentences.append(joined_tokens)
        self.sentences += proc_sentences
        return proc_sentences

    def read_sentences(self, news_sentences_file: str) -> List[List[str]]:
        with open(news_sentences_file, "r") as f:
            sentences = f.readlines()
        proc_sentences = []
        length = len(sentences)
        for i, sentence in enumerate(sentences):
            if (i + 1) % 3000 == 0: print(f"Processing sentence {i+1}/{length}")
            sent_clean = sentence.split("\t")[1]
            sent_nlp = self.nlp(sent_clean)
            joined_tokens = " ".join(
                [str(token).strip() for token in sent_nlp if len(str(token).strip()) > 0]
            )
            proc_sentences.append(joined_tokens)
        self.sentences += proc_sentences
        return proc_sentences

    def write_to_single_file(self, output_file: str) -> None:
        with open(output_file, "w") as f:
            for sentence in self.sentences:
                f.write(sentence)
                f.write("\n")


class Embeddings(object):

    def __init__(self):
        self.model: Word2Vec = None

    def train_w2v(self, sentences_file: str, save_model: str = None) -> Word2Vec:
        with open(sentences_file, "r") as f:
            lines = f.readlines()
        sentences = []
        for line in lines:
            sentences.append([token for token in line.lower().split()])
        self.model = Word2Vec(
            sentences,
            min_count=5,
            size=300,
            workers=3,
            window=5,
            iter=10,
            sg=0,  # CBOW
            seed=1338
        )
        if save_model: self.model.save(save_model)
        return self.model

    def finetune_w2v(self, sentences_file: str, save_model: str = None) -> Word2Vec:
        with open(sentences_file, "r") as f:
            lines = f.readlines()
        sentences = []
        for line in lines:
            sentences.append([token for token in line.lower().split()])
        self.model.train(sentences=sentences, total_examples=len(sentences), epochs=5)
        if save_model: self.model.save(save_model)
        return self.model

    def load_w2v(self, model_file: str) -> Word2Vec:
        self.model = Word2Vec.load(model_file)
        return self.model


def combine_news_txts():
    news_proc = NewsPreprocessor()
    print("Reading Kompas...")
    news_proc.read_news("../data/raw/news/kompas/txt/")
    print("Reading Tempo...")
    news_proc.read_news("../data/raw/news/tempo/txt/")
    print("Reading News Sentences...")
    news_proc.read_sentences("../data/raw/news/sentences.txt")
    print("Writing sentences...")
    news_proc.write_to_single_file("../data/processed/news.txt")


def train_news_w2v():
    emb = Embeddings()
    emb.train_w2v(
        "../data/processed/news.txt",
        save_model="../pretrain/embeddings/news.bin"
    )


def finetune_news_w2v():
    emb = Embeddings()
    emb.load_w2v("../pretrain/embeddings/id.bin")
    emb.finetune_w2v(
        "../data/processed/news.txt",
        save_model="../pretrain/embeddings/id_ft.bin"
    )


def test_pretrain_w2v():
    emb = Embeddings()
    emb.load_w2v("../pretrain/news_w2v.model")
    print(emb.model.wv.most_similar("reuters"))


if __name__ == "__main__":
    train_news_w2v()
