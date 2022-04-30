import json
import os
import pandas as pd

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, map_path):

        with open(map_path) as fp:
            icd9_map = json.load(fp)

        # compile data
        unq_codes = set()
        unq_cats = set()

        data_dict = dict()
        df = pd.read_csv(data_path, dtype=str)
        for row in df.to_dict('records'):
            try:
                patient = int(row['SUBJECT_ID'])
                visit = int(row['HADM_ID'])
                code_set = (int(row['SEQ_NUM']), row['ICD9_CODE'])  # allows for sorting
            except Exception as e:
                continue

            if patient not in data_dict.keys():
                data_dict[patient] = dict()
            if visit not in data_dict[patient].keys():
                data_dict[patient][visit] = list()
            data_dict[patient][visit].append(code_set)

        data_codes = list()
        data_categories = list()
        for i_patient, patient in enumerate(data_dict.keys()):
            patient_list = list()
            cat_list = list()
            if len(data_dict[patient].keys()) < 2:
                continue  # filter out patients with less than two visits
            sorted_visits = sorted(data_dict[patient].keys())
            for visit in sorted_visits[:-1]:
                codes = list()
                for seq, code in sorted(data_dict[patient][visit]):
                    unq_codes.add(code)
                    codes.append(code)
                patient_list.append(codes)

            cats = list()
            for seq, code in sorted(data_dict[patient][sorted_visits[-1]]):
                cat = icd9_map[code[:3] if code[0] != 'E' else code[:4]]
                unq_cats.add(cat)
                cats.append(cat)
            cat_list.append(list(set(cats)))

            data_codes.append(patient_list)
            data_categories.append(cat_list)

        num_visits = [len(patient) for patient in data_codes]
        num_codes = [len(visit) for patient in data_codes for visit in patient]
        num_categories = [len(visit) for patient in data_categories for visit in patient]
        self.num_patients = len(data_codes)
        self.max_num_visits = max(num_visits)
        self.max_num_codes = max(num_codes)
        self.max_num_categories = max(num_categories)

        self.idx2code = sorted(unq_codes)
        self.code2idx = {}
        for idx, code in enumerate(self.idx2code):
            self.code2idx[code] = idx

        self.idx2category = sorted(unq_cats)
        self.category2idx = {}
        for idx, cat in enumerate(self.idx2category):
            self.category2idx[cat] = idx

        self.x = data_codes
        self.y = data_categories

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]