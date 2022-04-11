import os
import pandas as pd
from icd9cms.icd9 import search

DATA_PATH = ".\\data"

cat_idx = 2

# compile data
unq_codes = set()
unq_cats = set()

data_dict = dict()
# for f in ['DIAGNOSES_ICD.csv', 'PROCEDURES_ICD.csv']:
for f in ['DIAGNOSES_ICD.csv']:
    df = pd.read_csv(os.path.join(DATA_PATH, f), dtype=str)
    for row in df.to_dict('records'):
        try:
            patient = int(row['SUBJECT_ID'])
            visit = int(row['HADM_ID'])
            code = row['ICD9_CODE'] if row['ICD9_CODE'] != '71970' else '7197'
            code_set = (int(row['SEQ_NUM']), code)  # allows for sorting
        except Exception as e:
            continue

        if patient not in data_dict.keys():
            data_dict[patient] = dict()
        if visit not in data_dict[patient].keys():
            data_dict[patient][visit] = list()
        data_dict[patient][visit].append(code_set)

data_codes = list()
data_cat = list()
for patient in data_dict.keys():
    patient_list = list()
    cat_list = list()
    if len(data_dict[patient].keys()) < 2:
        continue  # filter out patients with less than two visits
    for j_visit, visit in enumerate(sorted(data_dict[patient].keys())):
        codes = list()
        cats = list()
        for seq, code in sorted(data_dict[patient][visit]):
            try:
                cat = search(code).ancestors()[-2]
                unq_codes.add(code)
                codes.append(code)
                unq_cats.add(cat)
                cats.append(cat)
            except Exception as e:
                print(code, type(code), e)
                continue
        patient_list.append(codes)
        cat_list.append(list(set(cats)))

    data_codes.append(patient_list)
    data_cat.append(cat_list)

num_visits = [len(patient) for patient in data_codes]
num_codes = [len(visit) for patient in data_codes for visit in patient]
num_cats = [len(visit) for patient in data_cat for visit in patient]
num_patients = len(data_codes)
max_num_visits = max(num_visits)
max_num_codes = max(num_codes)
max_num_cats = max(num_cats)

print('number of patients', num_patients)
print('number of visits', sum(num_visits))
print('average number of visits', "{:.3f}".format(float(sum(num_visits)) / len(num_visits)))
print()
print('number of unique codes', len(unq_codes))
print('average number of codes per visit', "{:.2f}".format(float(sum(num_codes)) / len(num_codes)))
print('max number of codes per visit', max_num_codes)
print()
print('number of unique categories', len(unq_cats))
print('average number of categories per visit', "{:.2f}".format(float(sum(num_cats)) / len(num_cats)))
print('max number of categories per visit', max_num_cats)
print()
