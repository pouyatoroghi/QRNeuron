import os
import math
import logging
from functools import lru_cache
from tqdm import tqdm
import json
from pathlib import Path
from .neuron_attribution import NeuronAtrribution

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

class NaicaKeyNeuron:
        
        def __init__(
            self,
            model_name: str,
            data_samples: str,
            result_dir: str,
            attr_threshold: float = 0.2,
            common_threshold: float = 0.3,
            batch_size: int = 20,
            steps: int = 20,
            top_v: int = 10,
            option_letters: str = ["A", "B", "C", "D"],
            neuron_attr_score_file: str = "neuron_attr_score.json",
            common_neuron_file: str = "common_neuron.json",
            key_neuron_file: str = "key_neuron.json",
        ):
            """_summary_

            Args:
                model_name (str): model name on HuggingFace, e.g., meta-llama/Llama-2-7b-chat-hf. Recommend instruction-alined models
                data_samples (str): a json object contains queries. see the data format in /data/domain_sample_multi_choice_qa.json
                result_dir (str): result directory to store intermidiate files
                attr_threshold (float, optional):  the attribution threshold t to filter neurons with low scores. Defaults to 0.2.
                common_threshold (float, optional): the threshold to find common neurons. Defaults to 0.3.
                batch_size (int, optional): batch size. Defaults to 20.
                steps (int, optional): the number of the estimation steps to compute ingrated gradients. Defaults to 20.
                top_v (int, optional): we select top-v neurons with the highest score from the detected cluster. Defaults to 10.
                option_letters (str, optional): the option letter set. Defaults to ["A", "B", "C", "D"].
                neuron_attr_score_file (str, optional): the file to store neuron attribution. Defaults to "neuron_attr_score.json".
                common_neuron_file (str, optional): the file to store common neurons. Defaults to "common_neuron.json".
                key_neuron_file (str, optional): the file to store the extracted key neurons. Defaults to "key_neuron.json".
            """
            self.model_name = model_name
            self.data_samples = data_samples
            self.option_letters = option_letters
            self.result_dir = result_dir
            self.attr_threshold = attr_threshold
            self.common_threshold = common_threshold
            self.batch_size = batch_size
            self.steps = steps
            self.top_v = top_v
            
            
            self.neuron_attribution_file = Path(self.result_dir+neuron_attr_score_file)
            self.common_neuron_file = Path(self.result_dir+common_neuron_file)
            self.key_neuron_file = Path(self.result_dir+key_neuron_file)
            
            self.UUIDS = list()
            self.coares_neuron_attribution_scores = list()
            self.neuron_attribution = list()
            self.clusters = list()
            self.common_neurons = list()
            self.occurances = dict() 
            
            self.NA = NeuronAtrribution(model_name=self.model_name, option_letters=self.option_letters)
        
        def _cal_na(self):
            logging.info("start to calculate neuron attribution")
            
            wf = open(self.neuron_attribution_file, "w", encoding="utf8")
            logging.info("created file to store neuron attribution scores")
            
              
            INDICES = list(range(0, len(self.data_samples), 1))
            self.UUIDS = list(self.data_samples.keys())
            for i, idx in enumerate(tqdm(INDICES)):
                
                _uuid = self.UUIDS[idx]
                logging.info("processing the query {a} ......".format(a=_uuid))
                
                prompts, ground_truth, relation_name = (
                        self.data_samples[_uuid]["sentences"],
                        self.data_samples[_uuid]["obj_label"],
                    self.data_samples[_uuid]["relation_name"],
                )
                assert len(prompts) == len(ground_truth), "Must have equal number of queries and labels"
                
            
                neurons, attr_scores, neuron_freq = self.NA.get_neuron_attribution(
                    prompts=prompts,
                    ground_truths=ground_truth,
                    batch_size=self.batch_size,
                    steps=self.steps,
                    threshold=self.attr_threshold,
                    quiet=True,
                ) 
                    
                logging.info("for query {a}, we found {b} coarse neurons".format(a=_uuid, b=len(neurons)))
                tmp_attr_obj = {"uuid":_uuid, "neurons": neurons, "attr_scores":attr_scores, "neuron_freq":neuron_freq}
                self.coares_neuron_attribution_scores.append(tmp_attr_obj)
                json.dump(tmp_attr_obj, wf, ensure_ascii=False)
                wf.write("\n")
                wf.flush()
                logger.info("query {a} is finished. You check the result file".format(a=_uuid))

            # print(self.neuron_attribution_file)
            # for line in open(self.neuron_attribution_file):
            #     obj = json.loads(line)
            #     self.coares_neuron_attribution_scores.append(obj)
                
            print(len(self.coares_neuron_attribution_scores))
                
            self.clusters, all_attr_scores = list(), list()
            for attr_obj in self.coares_neuron_attribution_scores:
                name_uuid, neurons, attr_scores = attr_obj["uuid"], attr_obj["neurons"], attr_obj["attr_scores"]
                neuron_names = [str(neuron[0])+"_"+str(neuron[1]) for neuron in neurons]
                self.clusters.append(neuron_names)
                
                score_dict = dict()
                for i, neuron in enumerate(neurons):
                    neuron_name = str(neuron[0])+"_"+str(neuron[1])
                    score = attr_scores[i]
                    score_dict[neuron_name] = score
                    
                all_attr_scores.append(score_dict)
            
            
            for index, neurons in enumerate(self.clusters):
                scores = all_attr_scores[index]
                sum_values = sum(scores.values())
                temp_na = dict()
                for neuron in neurons:
                    if neuron not in self.occurances:
                        self.occurances[neuron] = 0
                    self.occurances[neuron] += 1
                    
                    if neuron not in temp_na:
                        temp_na[neuron] = scores[neuron] / sum_values if sum_values > 0 else 0
                self.neuron_attribution.append(temp_na)
            
            logger.info("the calulation of neuron attribution (NA) is finished, and the scores are stored in {a}".format(a=self.neuron_attribution_file))
        
        
        def _find_common_neuron(self):
            logger.info("start to find common neurons")
            cluster_num = len(self.clusters)
            _common_t = int(cluster_num * self.common_threshold)
            logger.info("the frequency u for obtaining common neurons is {a}".format(a=(self.common_threshold, _common_t)))
            temp_common_neurons = [e[0] for e in sorted(self.occurances.items(), key= lambda e:e[1], reverse=True) if e[1] > _common_t]
            temp_common_neurons = [e.split("_") for e in temp_common_neurons]
            temp_common_neurons = [[int(e[0]), int(e[1])] for e in temp_common_neurons]
            common_neuron_wf = open(self.common_neuron_file, "w")
            json.dump({"common_neuron": temp_common_neurons}, common_neuron_wf)
            logger.info("{a} common neurons are found!".format(a=len(temp_common_neurons)))
            self.common_neurons = temp_common_neurons
        
        def _extract_key_neuron(self):
            
            self._cal_na()
            self._find_common_neuron()
            
            naica_scores = list()
            for index, na_scores in enumerate(self.neuron_attribution):
                temp_naica_values = dict()
                for neuron, na in na_scores.items():
                    ica = math.log(len(self.clusters) / (self.occurances[neuron] + 1))
                    na_ica = na * ica
                    temp_naica_values[neuron] = na_ica
                naica_scores.append(temp_naica_values)


            kn_wf = open(self.common_neuron_file, "w")
            for index, naica_value in enumerate(naica_scores):
                
                neurons = sorted(naica_value.items(), key= lambda e:e[1], reverse=True)
                neurons = [e for e in neurons if e[1] > 0]
                key_neurons = [e for e in neurons][:self.top_v]
                key_neurons = [w for w in key_neurons if w[0] not in self.common_neurons]
                key_neuron_names = [w[0].split("_") for w in key_neurons]
                key_neuron_names = [[int(w[0]), int(w[1])] for w in key_neuron_names]
                scores = [w[1] for w in key_neurons]
                
                obj = {
                    "uuid": self.UUIDS[index],
                    "neurons": key_neuron_names,
                    "scores":scores
                }
                json.dump(obj, kn_wf)
                kn_wf.write("\n")
            
            avg_kn_num = sum([len(ns) for ns in key_neuron_names]) /  len(key_neuron_names) if len(key_neuron_names) !=0 else 0
            logger.info("finish! the avg key neuron number = {a}".format(a=avg_kn_num))





    
    



    

        