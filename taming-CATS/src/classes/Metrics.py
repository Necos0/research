import os
import uuid
from pathlib import Path
import textstat
import sacrebleu
import evaluate
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# COMET / LENS は重い評価指標。未インストールでも動くよう任意化（スモークテスト対応）
try:
    from comet.models import download_model as download_comet_model, load_from_checkpoint
    _COMET_AVAILABLE = True
except Exception:
    _COMET_AVAILABLE = False
try:
    from lens import download_model as download_lens_model, LENS
    _LENS_AVAILABLE = True
except Exception:
    _LENS_AVAILABLE = False

nltk.download('punkt')
nltk.download('punkt_tab')  # 新しい nltk は sent_tokenize に punkt_tab を要求

class Metrics:
    bertscore_model = None # on first use if ref provided
    comet_model = None 
    sari_model = None
    lens_model = None
    lens_unavailable = False

    def __init__(self, input_text, reference_text=None, source_text=None):
        self.text = input_text
        self.reference = reference_text
        self.source = source_text
        self.metrics = {}

        self.bertscore = 0.0 # comparison with the source
        self.bertscore_ref = 0.0 # comparison with a ref
        self.bleu = 0.0 # comparison with the source
        self.bleu_ref = 0.0 # comparison with the ref
        self.comet = 0.0
        self.sari = 0.0
        self.lens = None

    @staticmethod
    def load_bertscore():
        """Loads BERTScore model with unique experiment_id to avoid cache collisions."""
        # Always create a new instance with unique experiment_id for parallel jobs
        experiment_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        return evaluate.load("bertscore", experiment_id=experiment_id)

    @staticmethod
    def load_sari():
        """Loads SARI model once for efficiency."""
        if Metrics.sari_model is None:
            Metrics.sari_model = evaluate.load("sari")

    @staticmethod
    def load_comet():
        """Loads COMET model once for efficiency."""
        if Metrics.comet_model is None:
            model_path = download_comet_model("Unbabel/wmt22-comet-da")
            Metrics.comet_model = load_from_checkpoint(model_path)

    @staticmethod
    def load_lens():
        """Loads LENS model via lens-metric and caches it for reuse."""
        if not _LENS_AVAILABLE:
            Metrics.lens_unavailable = True
            return None
        if Metrics.lens_unavailable:
            return None

        if Metrics.lens_model is not None:
            return Metrics.lens_model

        try:
            lens_path = download_lens_model("davidheineman/lens")
            Metrics.lens_model = LENS(lens_path, rescale=True)
            return Metrics.lens_model
        except Exception as e:
            Metrics.lens_unavailable = True
            print(f"[WARN] LENS metric unavailable. Skipping LENS computation. Error: {e}")
            return None

    def count_words(self):
        return len(word_tokenize(self.text))

    def count_chars(self):
        return len(self.text)

    def count_alphanum(self):
        return sum(1 for char in self.text if char.isalpha())

    def count_sents(self):
        return len(sent_tokenize(self.text))

    def compute_fre(self):
        return textstat.flesch_reading_ease(self.text)

    def compute_ari(self):
        return textstat.automated_readability_index(self.text)

    def compute_fkgl(self):
        return textstat.flesch_kincaid_grade(self.text)

    def compute_dale_chall(self):
        return textstat.dale_chall_readability_score(self.text)

    def compute_bleu(self):
        """
        Computes BLEU score between the source and the target.
        Corpus version used to account for possible text-level alignement.
        """
        if not self.source:
            return 0.0
        self.bleu = sacrebleu.corpus_bleu(hypotheses=[self.text], references=[[self.source]]).score
        return self.bleu
    
    def compute_bleu_w_reference(self):
        """
        Computes BLEU score between the target and the refrence.
        Corpus version used to account for possible text-level alignement.
        """
        if not self.reference:
            return 0.0
        self.bleu_ref = sacrebleu.corpus_bleu(hypotheses=[self.text], references=[[self.reference]]).score
        return self.bleu_ref

    def compute_bertscore(self):
        """
        Computes BERTScore using evaluate library.
        Compares the source and the target.
        """
        if os.getenv("SKIP_BERTSCORE") or not self.source:
            return 0.0
        bertscore_model = self.load_bertscore()
        results = bertscore_model.compute(
            predictions=[self.text], references=[self.source], lang="en"
        )
        self.bertscore = results["f1"][0]  # get only first value, F1
        return self.bertscore
    
    def compute_bertscore_w_reference(self):
        """
        Computes BERTScore using evaluate library.
        Compares the target and the reference.
        """
        if os.getenv("SKIP_BERTSCORE") or not self.reference:
            return 0.0
        bertscore_model = self.load_bertscore()
        results = bertscore_model.compute(
            predictions=[self.text], references=[self.reference], lang="en"
        )
        self.bertscore_ref = results["f1"][0]  # get only first value, F1
        return self.bertscore_ref

    def compute_comet(self):
        """Computes COMET score."""
        if not _COMET_AVAILABLE or not self.reference:
            return 0.0
        self.load_comet()
        data = [{"src": self.source, "mt": self.text, "ref": self.reference}]
        scores = Metrics.comet_model.predict(data)
        self.comet_ref = scores["scores"][0]  # get only first value
        return self.comet_ref

    def compute_sari(self):
        """Computes SARI score."""
        if not self.reference or not self.source:
            return 0.0  # SARI requires source (original) and reference (target)
        self.load_sari()
        source = [self.source]
        prediction = [self.text]
        reference = [[self.reference]] # NB expects a list of list
        sari_score = Metrics.sari_model.compute(sources=source, predictions=prediction, references=reference)
        # score = sari.corpus_score([self.text], [[self.reference]], [self.source])
        self.sari_ref = sari_score["sari"]  # Extract SARI score
        return self.sari_ref

    def compute_lens(self):
        """Computes LENS score for text simplification quality."""
        if not self.source:
            return None

        lens_model = self.load_lens()
        if lens_model is None:
            return None

        references = [[self.reference]] if self.reference else [[]]
        score_kwargs = {
            "batch_size": 8,
        }
        if torch.cuda.is_available():
            score_kwargs["devices"] = [0]

        try:
            scores = lens_model.score(
                [self.source],
                [self.text],
                references,
                **score_kwargs,
            )
        except TypeError:
            # Older versions may not accept devices.
            scores = lens_model.score(
                [self.source],
                [self.text],
                references,
                batch_size=8,
            )
        except Exception:
            return None

        if not scores:
            return None

        self.lens = float(scores[0])
        return self.lens

    def compute_metrics(self):
        """Computes all required metrics and returns them as a dictionary."""
        self.metrics = {
            'word_count': self.count_words(),
            'char_count': self.count_chars(),
            'alphanum_count': self.count_alphanum(),
            'sent_count': self.count_sents(),
            'FRE': self.compute_fre(),
            'ARI': self.compute_ari(),
            'FKGL': self.compute_fkgl(),
            'Dale-Chall': self.compute_dale_chall(),
            'BLEU': self.compute_bleu(),
            'BERTScore': self.compute_bertscore(),
            'BLEU_ref': self.compute_bleu_w_reference(),
            'BERTScore_ref': self.compute_bertscore_w_reference(),
            'COMET': self.compute_comet(),
            'SARI': self.compute_sari(),
            'LENS': self.compute_lens(),
        }

        return self.metrics

# prediction = "This is a whale."
# reference = "That is a killer whale which is more like a dolphin, actually."
# source = "This is a bird."
# metrics = Metrics(input_text=prediction, reference_text=reference, source_text=source)
# print(metrics.text)
# print(metrics.compute_metrics())
# print(f"BLEU: {metrics.compute_bleu()}")
# print(f"BLEU ref: {metrics.compute_bleu_w_reference()}")
# print(f"BERTScore: {metrics.compute_bertscore()}")
# print(f"BERTScore ref: {metrics.compute_bertscore_w_reference()}")

