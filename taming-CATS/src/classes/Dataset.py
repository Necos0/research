import os
import pandas as pd


class TSDataset:
    def __init__(self):
        self.data_df = pd.DataFrame()


class Newsela(TSDataset):
    def __init__(self, limit=None):
        super().__init__()
        self.data_dir = "./../../datasets/newsela/newsela_article_corpus_2016-01-29/articles"
        self.metadata_path = "./../../datasets/newsela/newsela_article_corpus_2016-01-29/articles_metadata.csv"
        self.limit = limit # None as default, else select int to slice
        # dataset metainfo
        self.dataset_name = "newsela"
        self.domain = "news"
        self.annotation = "human"
        self.alignment_level = "text"

        # relevant columns
        self.grade_level = "grade_level"
        self.simplification_version = "version"
        self.text = "content"
        self.language = "language"
        self.grouping_tag = "slug"

    def load_data(self):
        """ Loads Newsela onto a pandas df. """
        metadata = pd.read_csv(self.metadata_path, sep=",")
        data = []
        loaded_files = 0

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    data.append({"filename": filename, "content": text})
                    loaded_files += 1
                    if self.limit and loaded_files >= self.limit:
                        break

        articles_df = pd.DataFrame(data)
        self.data_df = pd.merge(articles_df, metadata, on="filename", how="inner")

    
class MedEASi(TSDataset):
    def __init__(self, limit=None):
        super().__init__()
        self.file_path = "./../../datasets/Med-EASi/Med-EASi.full.ori.csv"
        self.limit = limit # None as default, else select int to slice
        # dataset metainfo
        self.dataset_name = "medeasi"
        self.domain = "medical"
        self.annotation = "human"
        self.alignment_level = "sentence"
        self.language = "en"

        # relevant columns
        self.source_text = "Expert"
        self.target_text = "Simple"
        self.split = "split"
        self.grouping_tag = "idx"

    def load_data(self):
        """ Loads Med-EASi onto a pandas df. """
        try:
            data = pd.read_csv(self.file_path, sep=",")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        if self.limit:
            data = data.head(self.limit)

        self.data_df = data


class WikiLarge(TSDataset):
    def __init__(self, limit=None):
        super().__init__()
        self.data_dir = "./../../datasets/wiki/wikilarge"
        self.base_filename = "wiki.full.aner.ori."
        self.limit = limit

        # Dataset metainfo
        self.dataset_name = "wikilarge_ori"
        self.domain = "general"
        self.annotation = "automatic"
        self.alignment_level = "sentence"
        self.language = "en"

        # Relevant columns
        self.source_text = "source"
        self.target_text = "target"
        self.split = "split"
        self.grouping_tag = "idx"

    def load_data(self):

        splits = ["train", "valid", "test"]
        all_data = []

        for split in splits:
            src_path = os.path.join(self.data_dir, f"{self.base_filename}{split}.src")
            dst_path = os.path.join(self.data_dir, f"{self.base_filename}{split}.dst")

            # check if the files exist
            if not os.path.exists(src_path) or not os.path.exists(dst_path):
                print(f"Warning: Missing files for {split} split.")
                continue

            with open(src_path, "r", encoding="utf-8") as src_file, open(dst_path, "r", encoding="utf-8") as dst_file:
                src_lines = src_file.readlines()
                dst_lines = dst_file.readlines()

            # Ensure same number of lines in source and target
            if len(src_lines) != len(dst_lines):
                print(f"Warning: Mismatched line counts in {split} split.")

            # to df with cols source, target and split
            split_df = pd.DataFrame({
                "source": src_lines,
                "target": dst_lines,
                "split": split
            })

            # for dev
            if self.limit:
                split_df = split_df.head(self.limit)

            all_data.append(split_df)

        # combine into one df
        if all_data:
            self.data_df = pd.concat(all_data, ignore_index=True)
            self.data_df['idx'] = pd.Series(range(1, len(self.data_df) + 1), index=self.data_df.index)
        else:
            print("Error: No data loaded.")

class SimPA(TSDataset):
    """ 
    Parent class for the SimPA dataset.
    Includes children classes for the lexical and syntactic simplification subsets.
    """
    def __init__(self, data_dir="./../../datasets/simpa", limit=None):
        super().__init__()
        self.data_dir = data_dir
        self.limit = limit

        # Dataset metainfo
        self.domain = "public_administration"
        self.annotation = "human"
        self.alignment_level = "sentence"
        self.language = "en"

        # Relevant columns
        self.text = "content"
        self.simplification_version = "version"
        self.grouping_tag = "slug"

    def load_data(self):
        raise NotImplementedError("Subclasses must implement this method")

class SimPALex(SimPA):
    def __init__(self, limit=None):
        super().__init__(limit=limit)
        self.dataset_name = "simpa_lexical"

    def load_data(self):
        ls_orig_path = os.path.join(self.data_dir, "ls.original")
        ls_simp_path = os.path.join(self.data_dir, "ls.simplified")
        
        # check the files exist
        if not os.path.exists(ls_orig_path) or not os.path.exists(ls_simp_path):
            print("Warning: Files missing.")
            return

        with open(ls_orig_path, "r", encoding="utf-8") as f_orig, open(ls_simp_path, "r", encoding="utf-8") as f_simp:
            ls_orig_lines = f_orig.readlines()
            ls_simp_lines = f_simp.readlines()
        
        # double-check correct alignment
        # there are 3 identical lines for each source sentence
        # to align with 3 simplifications per source sentence
        if len(ls_orig_lines) % 3 != 0 or len(ls_orig_lines) != len(ls_simp_lines):
            print("Warning: source and target datasets differ in size.")
            return
        
        data = []
        group_id = 1
        for i in range(0, len(ls_orig_lines), 3):
            original_sentence = ls_orig_lines[i].strip()
            data.append((original_sentence, group_id, 0))  # Original sentence version 0
            for j in range(3):
                simplified_sentence = ls_simp_lines[i + j].strip()
                data.append((simplified_sentence, group_id, j + 1))  # Simplifications version 1,2,3
            group_id += 1
        
        df_ls = pd.DataFrame(data, columns=["content", "slug", "version"])
        
        # apply limit if specified (for dev only)
        if self.limit:
            df_ls = df_ls.head(self.limit)
        
        self.data_df = df_ls

class SimPASyn(SimPA):
    def __init__(self, limit=None):
        super().__init__(limit=limit)
        self.dataset_name = "simpa_syntactic"

    def load_data(self):
        ss_orig_path = os.path.join(self.data_dir, "ss.original")
        ss_simp_path = os.path.join(self.data_dir, "ss.simplified")
        
        # check the files exist
        if not os.path.exists(ss_orig_path) or not os.path.exists(ss_simp_path):
            print("Warning: Files missing.")
            return

        with open(ss_orig_path, "r", encoding="utf-8") as f_orig, open(ss_simp_path, "r", encoding="utf-8") as f_simp:
            ss_orig_lines = f_orig.readlines()
            ss_simp_lines = f_simp.readlines()
        
        # double-check correct alignment
        if len(ss_orig_lines) != len(ss_simp_lines):
            print("Warning: source and target datasets differ in size.")
            return
        
        data = []
        for i in range(len(ss_orig_lines)):
            original_sentence = ss_orig_lines[i].strip()
            syntactically_simplified_sentence = ss_simp_lines[i].strip()
            
            data.append((original_sentence, i + 1, 0))  # Original sentence version 0
            data.append((syntactically_simplified_sentence, i + 1, 1))  # Syntactic simplification version 1
        
        df_ss = pd.DataFrame(data, columns=["content", "slug", "version"])
        
        # apply limit if specified (for dev only)
        if self.limit:
            df_ss = df_ss.head(self.limit)
        
        self.data_df = df_ss



# dataset = SimPASyn(limit=10)
# dataset.load_data()
# print(dataset.data_df)


# dataset = SimPALex(limit=10)
# dataset.load_data()
# print(dataset.data_df)


# dataset = WikiLarge(limit=10)
# dataset.load_data()
# print(dataset.data_df)


# dataset = MedEASi(limit=10)
# dataset.load_data()
# print(dataset.data_df)


# dataset = Newsela(limit=10)
# dataset.load_data()
# print(dataset.data_df)



