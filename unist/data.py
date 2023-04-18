import json
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == "-lrb-"):
        return "("
    elif (token.lower() == "-rrb-"):
        return ")"
    elif (token.lower() == "-lsb-"):
        return "["
    elif (token.lower() == "-rsb-"):
        return "]"
    elif (token.lower() == "-lcb-"):
        return "{"
    elif (token.lower() == "-rcb-"):
        return "}"
    return token


class TACREDDataset(Dataset):
    def __init__(self, data_file, no_task_desc=False, use_pseudo=False, raw_labelset=None, mask_entity=False, mask_token=None):
        self.data = []
        if raw_labelset is None:
            raw_labelset = ["no_relation", "per:title", "org:top_members/employees", "per:employee_of", "org:alternate_names", "org:country_of_headquarters", "per:countries_of_residence", "org:city_of_headquarters", "per:cities_of_residence", "per:age", "per:stateorprovinces_of_residence", "per:origin", "org:subsidiaries", "org:parents", "per:spouse", "org:stateorprovince_of_headquarters", "per:children", "per:other_family", "per:alternate_names", "org:members", "per:siblings", "per:schools_attended", "per:parents", "per:date_of_death", "org:member_of", "org:founded_by", "org:website", "per:cause_of_death", "org:political/religious_affiliation", "org:founded", "per:city_of_death", "org:shareholders", "org:number_of_employees/members", "per:date_of_birth", "per:city_of_birth", "per:charges", "per:stateorprovince_of_death", "per:religion", "per:stateorprovince_of_birth", "per:country_of_birth", "org:dissolved", "per:country_of_death"]
        self.labelset = [self.preprocess_label(label) for label in raw_labelset]

        with open(data_file, "r") as f:
            data = json.load(f)
        
        self.mask_entity = mask_entity
        self.mask_token = mask_token

        # 是否使用伪数据
        self.use_pseudo = use_pseudo
        # 是否使用任务描述：task_desc

        for d in tqdm(data, desc="Preprocessing"):
            # if len(d) == 2:
            if self.use_pseudo:
                # assert self.use_pseudo
                features = []
                for item in d:
                    feature = self.get_feature(item, no_task_desc)
                    features.append(feature)
                self.data.append(features)
            else:
                feature = self.get_feature(d, no_task_desc)
                self.data.append(feature)

    def get_feature(self, d, no_task_desc):
        ss, se, st = d["subj_start"], d["subj_end"]+1, d["subj_type"].lower()
        os, oe, ot = d["obj_start"], d["obj_end"]+1, d["obj_type"].lower()

        tokens = [convert_token(token) for token in d["token"]]
        label = d["relation"]
        label = self.preprocess_label(label) # 处理标签
        assert label in self.labelset
        desc_ss = desc_se = desc_os = desc_oe = None

        if self.mask_entity:
            tokens[ss:se] = [self.mask_token] * (se - ss)
            tokens[os:oe] = [self.mask_token] * (oe - os)
        
        # add marker and type (and task description if needed)
        if no_task_desc: # 不使用任务描述
            if ss < os:
                sent = tokens[:ss] + ["<SUBJ>"] + [st] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:os] + ["<OBJ>"] + [ot] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:]
            else:
                sent = tokens[:os] + ["<OBJ>"] + [ot] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:ss] + ["<SUBJ>"] + [st] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:]            
        else: # 使用任务描述
            if ss < os:
                sent = tokens[:ss] + ["<SUBJ>"] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:os] + ["<OBJ>"] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:]
            else:
                sent = tokens[:os] + ["<OBJ>"] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:ss] + ["<SUBJ>"] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:]                
            # 加上任务描述
            # sent += ["</s>", "</s>", "Describe", "the", "relationship", "between"] + [st] + tokens[ss:se] + ["and"] + [ot] + tokens[os:oe] + ["."]
            sent += ["</s>", "</s>", "Describe", "the", "relationship", "between"]
            desc_ss = len(sent)
            sent += [st] + tokens[ss:se]
            desc_se = len(sent) - 1
            sent += ["and"]
            desc_os = len(sent) 
            sent += [ot] + tokens[os:oe]
            desc_oe = len(sent) - 1
            sent += ["."]
        subj_start, subj_end = "<SUBJ>", "</SUBJ>"
        obj_start, obj_end = "<OBJ>", "</OBJ>"
        # 计算插入特殊符号之后的实体开始和结束位置
        new_ss, new_se = sent.index(subj_start), sent.index(subj_end)
        new_os, new_oe = sent.index(obj_start), sent.index(obj_end)
        return [sent, label, new_ss, new_se, new_os, new_oe, desc_ss, desc_se, desc_os, desc_oe]

    def preprocess_label(self, label):
        rep_rule = (("_", " "), ("per:", "person "), ("org:", "organization "), ("stateorprovince", "state or province"))
        for r in rep_rule:
            label = label.replace(*r)
        return label

    def get_neg_label(self, feature):
        pos = feature[1] # 获取pos label
        neg = pos # 寻找到第一个neg label
        while neg == pos:
            neg = random.choice(self.labelset)
        return neg
        # return feature[:2] + [neg] + feature[2:] # sent, pos, neg, ss, se, os, oe

    def __getitem__(self, idx):
        if not self.use_pseudo: # 加上一个负标签
            feature = self.data[idx]
            neg = self.get_neg_label(feature)
            return feature[:2] + [neg] + feature[2:]
        else: # 使用伪数据，随机选择一个
            features = self.data[idx]
            feature1 = features[0]
            rand_idx = np.random.randint(1, len(features))
            feature2 = features[rand_idx]
            return [feature[:2] + [self.get_neg_label(feature)] + feature[2:] for feature in [feature1, feature2]]
            # return [self.add_neg_label(self.data[idx][0]), self.add_neg_label(self.data[idx][1])]
    
    def __len__(self):
        return len(self.data)    

class RETACREDDataset(TACREDDataset):
    def __init__(self, data_file, no_task_desc=False, use_pseudo=False, raw_labelset=None, mask_entity=False, mask_token=None):
        # 定义自己的raw_labelset
        raw_labelset = ['no_relation', 'org:website', 'per:country_of_death', 'per:cause_of_death', 'per:children', 'per:origin', 'org:political/religious_affiliation', 'per:cities_of_residence', 'per:title', 'per:charges', 'per:religion', 'org:number_of_employees/members', 'per:city_of_death', 'per:city_of_birth', 'per:countries_of_residence', 'org:top_members/employees', 'org:city_of_branch', 'per:parents', 'per:employee_of', 'per:stateorprovince_of_birth', 'org:country_of_branch', 'org:dissolved', 'per:date_of_death', 'org:founded', 'per:age', 'per:country_of_birth', 'org:members', 'per:spouse', 'org:founded_by', 'per:date_of_birth', 'per:identity', 'per:stateorprovinces_of_residence', 'org:alternate_names', 'org:shareholders', 'org:member_of', 'per:schools_attended', 'org:stateorprovince_of_branch', 'per:other_family', 'per:siblings', 'per:stateorprovince_of_death']
        super().__init__(data_file, no_task_desc, use_pseudo, raw_labelset, mask_entity, mask_token)
        

class UFETDataset(Dataset):
    def __init__(self, data_file, label_file, no_duplicates, no_task_desc=False):
        self.data = []
        with open(label_file, "r") as f:
            raw_labelset = f.read().splitlines()
        self.labelset = raw_labelset
            
        with open(data_file, "r") as f:
            data = [json.loads(line) for line in f.read().splitlines()]
        
        for d in tqdm(data, desc="Preprocessing"):
            left_tokens = [convert_token(token) for token in d["left_context_token"]]
            right_tokens = [convert_token(token) for token in d["right_context_token"]]
            entity = [convert_token(token) for token in d["mention_span"].split()]

            # preprocess label
            labels = [self.preprocess_label(label) for label in d["y_str"]]
            assert all([label in self.labelset for label in labels])

            sent = left_tokens + ["<E>"] + entity + ["</E>"] + right_tokens
            if not no_task_desc:
                sent += ["</s>", "</s>", "Describe", "the", "type", "of"] + entity + ["."]

            if no_duplicates:
                self.data.append([sent, "", labels])
            else:
                for label in labels:                
                    self.data.append([sent, label, labels])

    def preprocess_label(self, label):
        label = label.replace("_", " ")
        return label

    def __getitem__(self, idx):
        pos = self.data[idx][1]
        neg = pos
        all_pos = self.data[idx][2]
        while neg in all_pos:
            neg = random.choice(self.labelset)
        return self.data[idx][:2] + [neg] + self.data[idx][2:]

    def __len__(self):
        return len(self.data)
        

class MAVENDataset(Dataset):
    def __init__(self, data_file, no_task_desc=False):
        self.data = []
        raw_labelset = ["Know", "Warning", "Catastrophe", "Placing", "Causation", "Arriving", "Sending", "Protest", "Preventing_or_letting", "Motion", "Damaging", "Destroying", "Death", "Perception_active", "Presence", "Influence", "Receiving", "Check", "Hostile_encounter", "Killing", "Conquering", "Releasing", "Attack", "Earnings_and_losses", "Choosing", "Traveling", "Recovering", "Using", "Coming_to_be", "Cause_to_be_included", "Process_start", "Change_event_time", "Reporting", "Bodily_harm", "Suspicion", "Statement", "Cause_change_of_position_on_a_scale", "Coming_to_believe", "Expressing_publicly", "Request", "Control", "Supporting", "Defending", "Building", "Military_operation", "Self_motion", "GetReady", "Forming_relationships", "Becoming_a_member", "Action", "Removing", "Surrendering", "Agree_or_refuse_to_act", "Participation", "Deciding", "Education_teaching", "Emptying", "Getting", "Besieging", "Creating", "Process_end", "Body_movement", "Expansion", "Telling", "Change", "Legal_rulings", "Bearing_arms", "Giving", "Name_conferral", "Arranging", "Use_firearm", "Committing_crime", "Assistance", "Surrounding", "Quarreling", "Expend_resource", "Motion_directional", "Bringing", "Communication", "Containing", "Manufacturing", "Social_event", "Robbery", "Competition", "Writing", "Rescuing", "Judgment_communication", "Change_tool", "Hold", "Being_in_operation", "Recording", "Carry_goods", "Cost", "Departing", "GiveUp", "Change_of_leadership", "Escaping", "Aiming", "Hindering", "Preserving", "Create_artwork", "Openness", "Connect", "Reveal_secret", "Response", "Scrutiny", "Lighting", "Criminal_investigation", "Hiding_objects", "Confronting_problem", "Renting", "Breathing", "Patrolling", "Arrest", "Convincing", "Commerce_sell", "Cure", "Temporary_stay", "Dispersal", "Collaboration", "Extradition", "Change_sentiment", "Commitment", "Commerce_pay", "Filling", "Becoming", "Achieve", "Practice", "Cause_change_of_strength", "Supply", "Cause_to_amalgamate", "Scouring", "Violence", "Reforming_a_system", "Come_together", "Wearing", "Cause_to_make_progress", "Legality", "Employment", "Rite", "Publishing", "Adducing", "Exchange", "Ratification", "Sign_agreement", "Commerce_buy", "Imposing_obligation", "Rewards_and_punishments", "Institutionalization", "Testing", "Ingestion", "Labeling", "Kidnapping", "Submitting_documents", "Prison", "Justifying", "Emergency", "Terrorism", "Vocalizations", "Risk", "Resolve_problem", "Revenge", "Limiting", "Research", "Having_or_lacking_access", "Theft", "Incident", "Award"]
        self.labelset = [self.preprocess_label(label) for label in raw_labelset]
            
        with open(data_file, "r") as f:
            data = [json.loads(line) for line in f.read().splitlines()]
        
        for d in tqdm(data, desc="Preprocessing"):
            content = d["content"]
            for event in d["events"]:
                label = event["type"]
                label = self.preprocess_label(label)
                assert label in self.labelset

                for mention in event["mention"]:                 
                    sent_id = mention["sent_id"]
                    s, e = mention["offset"]
                    tokens = [convert_token(token) for token in content[sent_id]["tokens"]]
            
                    sent = tokens[:s] + ["<T>"] + tokens[s:e] + ["</T>"] + tokens[e:]
                    if not no_task_desc:
                        sent += ["</s>", "</s>", "Describe", "the", "type", "of"] + tokens[s:e] + ["."]       

                    self.data.append([sent, label]) 

    def preprocess_label(self, label):
        label = label.lower()
        label = label.replace("_", " ")
        return label

    def __getitem__(self, idx):
        pos = self.data[idx][1]
        neg = pos
        while neg == pos:
            neg = random.choice(self.labelset)
        return self.data[idx] + [neg]

    def __len__(self):
        return len(self.data)

class MAVENTestDataset(Dataset):
    def __init__(self, data_file, no_task_desc=False):
        self.data = []
        self.negative_trigger = defaultdict(list)

        raw_labelset = ["Know", "Warning", "Catastrophe", "Placing", "Causation", "Arriving", "Sending", "Protest", "Preventing_or_letting", "Motion", "Damaging", "Destroying", "Death", "Perception_active", "Presence", "Influence", "Receiving", "Check", "Hostile_encounter", "Killing", "Conquering", "Releasing", "Attack", "Earnings_and_losses", "Choosing", "Traveling", "Recovering", "Using", "Coming_to_be", "Cause_to_be_included", "Process_start", "Change_event_time", "Reporting", "Bodily_harm", "Suspicion", "Statement", "Cause_change_of_position_on_a_scale", "Coming_to_believe", "Expressing_publicly", "Request", "Control", "Supporting", "Defending", "Building", "Military_operation", "Self_motion", "GetReady", "Forming_relationships", "Becoming_a_member", "Action", "Removing", "Surrendering", "Agree_or_refuse_to_act", "Participation", "Deciding", "Education_teaching", "Emptying", "Getting", "Besieging", "Creating", "Process_end", "Body_movement", "Expansion", "Telling", "Change", "Legal_rulings", "Bearing_arms", "Giving", "Name_conferral", "Arranging", "Use_firearm", "Committing_crime", "Assistance", "Surrounding", "Quarreling", "Expend_resource", "Motion_directional", "Bringing", "Communication", "Containing", "Manufacturing", "Social_event", "Robbery", "Competition", "Writing", "Rescuing", "Judgment_communication", "Change_tool", "Hold", "Being_in_operation", "Recording", "Carry_goods", "Cost", "Departing", "GiveUp", "Change_of_leadership", "Escaping", "Aiming", "Hindering", "Preserving", "Create_artwork", "Openness", "Connect", "Reveal_secret", "Response", "Scrutiny", "Lighting", "Criminal_investigation", "Hiding_objects", "Confronting_problem", "Renting", "Breathing", "Patrolling", "Arrest", "Convincing", "Commerce_sell", "Cure", "Temporary_stay", "Dispersal", "Collaboration", "Extradition", "Change_sentiment", "Commitment", "Commerce_pay", "Filling", "Becoming", "Achieve", "Practice", "Cause_change_of_strength", "Supply", "Cause_to_amalgamate", "Scouring", "Violence", "Reforming_a_system", "Come_together", "Wearing", "Cause_to_make_progress", "Legality", "Employment", "Rite", "Publishing", "Adducing", "Exchange", "Ratification", "Sign_agreement", "Commerce_buy", "Imposing_obligation", "Rewards_and_punishments", "Institutionalization", "Testing", "Ingestion", "Labeling", "Kidnapping", "Submitting_documents", "Prison", "Justifying", "Emergency", "Terrorism", "Vocalizations", "Risk", "Resolve_problem", "Revenge", "Limiting", "Research", "Having_or_lacking_access", "Theft", "Incident", "Award"]
        self.labelset = [self.preprocess_label(label) for label in raw_labelset]
            
        with open(data_file, "r") as f:
            data = [json.loads(line) for line in f.read().splitlines()]
        
        for d in tqdm(data, desc="Preprocessing"):
            tokens = [convert_token(token) for token in d["tokens"]]
            s, e = d["span"]
            
            sent = tokens[:s] + ["<T>"] + tokens[s:e] + ["</T>"] + tokens[e:]
            if not no_task_desc:
                sent += ["</s>", "</s>", "Describe", "the", "type", "of"] + tokens[s:e] + ["."]  
                        
            if d["identify_infer"] == 0:
                # identifier thinks its not a event
                self.negative_trigger[d["docid"]].append({"id": d["id"], "type_id": 0})
            else:
                self.data.append([sent, d["docid"], d["id"]])

    def preprocess_label(self, label):
        label = label.lower()
        label = label.replace("_", " ")
        return label

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



class FewRelDataset(Dataset):
    def __init__(self, data_file, label_file, no_task_desc=False):
        self.data = []
        with open(label_file, "r") as f:
            raw_labelset = f.read().splitlines()
        self.labelset = [self.preprocess_label(label) for label in raw_labelset]

        with open(data_file, "r") as f:
            data = [json.loads(line) for line in f.read().splitlines()]

        for d in tqdm(data, desc="Preprocessing"):
            ss, se = d["head_span"]
            os, oe = d["tail_span"]

            tokens = [convert_token(token) for token in d["tokens"]]
            label = d["label"]
            label = self.preprocess_label(label)
            assert label in self.labelset
            
            # add marker and type (and task description if needed)
            if no_task_desc:
                if ss < os:
                    sent = tokens[:ss] + ["<SUBJ>"] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:os] + ["<OBJ>"] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:]
                else:
                    sent = tokens[:os] + ["<OBJ>"] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:ss] + ["<SUBJ>"] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:]            
            else:
                if ss < os:
                    sent = tokens[:ss] + ["<SUBJ>"] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:os] + ["<OBJ>"] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:]
                else:
                    sent = tokens[:os] + ["<OBJ>"] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:ss] + ["<SUBJ>"] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:]             
                sent += ["</s>", "</s>", "Describe", "the", "relationship", "between"] + tokens[ss:se] + ["and"] + tokens[os:oe] + ["."]

            self.data.append([sent, label])

    def preprocess_label(self, label):
        return label

    def __getitem__(self, idx):
        pos = self.data[idx][1]
        neg = pos
        while neg == pos:
            neg = random.choice(self.labelset)
        return self.data[idx] + [neg]
    
    def __len__(self):
        return len(self.data)  


if __name__ == "__main__":
    dataset = TACREDDataset('data/tacred/train.json', use_pseudo=False)
    sent, pos, neg, ss, se, os, oe, desc_ss, desc_se, desc_os, desc_oe = dataset[0]
    print(sent)
    print(pos)
    print(neg)
    print(sent[ss:se+1])
    print(sent[os:oe+1])
    print(sent[desc_ss:desc_se+1])
    print(sent[desc_os:desc_oe+1])
    print('-' * 100)

    # 测试tokenizer的方法
    # from transformers import RobertaTokenizer
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    # sent_inputs = tokenizer([sent, sent], padding=True, truncation=True, max_length=128, return_tensors="pt", is_split_into_words=True)
    # print(type(sent_inputs))
    # print(dir(sent_inputs))
    # print(sent_inputs)    # print(sent_inputs)