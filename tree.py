import numpy as np
import math

class Tree:
    def __init__(self, train_data =  None, max_depth = None):
        if train_data is None:
            raise Exception("Training data not given")
        
        self.train_data = train_data
        self.has_trained = False 
        self.root_node = None
        self.max_depth = max_depth
        
    def train(self):
        self.has_trained = True
        root_node = Node(is_root=True)
        depth = 0
        self.make_subtree(self.train_data, root_node, depth, 1, 1)
        self.root_node = root_node
        

    def make_subtree(self, training_instances, parent, depth, gain_ratio, entropy):
        candidate_splits = self.det_candidate_splits(training_instances)
        stopping_criteria = self._check_stopping_criteria(training_instances, gain_ratio, entropy, depth)

        parent.candidate_splits = candidate_splits

        if stopping_criteria:
            y_vals = training_instances[:, -1]
            class_1_count = len(y_vals[y_vals == 1])
            class_0_count = len(y_vals[y_vals == 0])
            if class_1_count >= class_0_count:
                parent.pred_value = 1
            else:
                parent.pred_value = 0

            parent.training_points.append(list(training_instances))
            return parent
        else:
            (gain_ratio, left_values, left_entropy, right_values, right_entropy, feature_index, criteria) = self.find_best_split(training_instances, candidate_splits)
            parent.split_cutoff = criteria
            parent.split_var = feature_index
            left_node = Node()
            right_node = Node()
            
            parent.left_node = self.make_subtree(left_values, left_node, depth+1, gain_ratio, left_entropy)
            parent.right_node = self.make_subtree(right_values, right_node, depth+1, gain_ratio, right_entropy)
        
        return parent

    def _check_stopping_criteria(self, training_instances, gain_ratio, entropy, depth):
        if len(training_instances) == 0 or entropy == 0 or gain_ratio == 0:
            if self.max_depth is not None and depth == self.max_depth:
                return False
            else:
                return True
        else:
            return False

    def det_candidate_splits(self, training_instances):
        feature_1_splits = self.det_candidate_splits_feature(training_instances[:, [0, -1]])
        feature_2_splits = self.det_candidate_splits_feature(training_instances[:, [1, -1]])

        return (feature_1_splits, feature_2_splits)

    
    def det_candidate_splits_feature(self, training_instances_feature):
        splits = set()
        sorted_training = training_instances_feature[training_instances_feature[:, 0].argsort()]
        labels = list(sorted_training[:, -1])
        values = list(sorted_training[:, 0])

        for index, value in enumerate(values):
            if (index+1) >= len(labels):
                break

            if labels[index] != labels[index + 1]:
                splits.add(value)

        return list(splits)

    def find_best_split(self, training_instances, feature_splits):
        gain_ratios = ([],[])

        for feature_index, feature_split in enumerate(feature_splits):
            for candidate_index, candidate_split in enumerate(feature_split):
                total_entropy = self._calc_entropy(training_instances)
                
                left_path_rows = training_instances[training_instances[:, feature_index] <= candidate_split]
                right_path_rows = training_instances[training_instances[:, feature_index] > candidate_split]

                if len(left_path_rows) == 0:
                    left_entropy = 0
                else:
                    left_entropy = self._calc_entropy(left_path_rows)
                
                if len(right_path_rows) == 0:
                    right_entropy = 0
                else:
                    right_entropy = self._calc_entropy(right_path_rows)

                left_weight = len(left_path_rows)/len(training_instances)
                right_weight = len(right_path_rows)/len(training_instances)

                split_entropy = left_weight*left_entropy + right_weight*right_entropy

                info_gain = total_entropy - split_entropy
                if split_entropy != 0:
                    info_gain_ratio = info_gain/split_entropy
                else: 
                    info_gain_ratio = 1

                gain_ratios[feature_index].append((info_gain_ratio, left_path_rows, left_entropy, right_path_rows, right_entropy, feature_index, candidate_split))
        
        if len(gain_ratios[0]) != 0 and len(gain_ratios[1]) != 0:
            feature_0_best_gain = sorted(gain_ratios[0], key=lambda x: x[0])[-1]
            feature_1_best_gain = sorted(gain_ratios[1], key=lambda x: x[0])[-1]
            
            if feature_0_best_gain[0] >= feature_1_best_gain[0]:
                return feature_0_best_gain
            else:
                return feature_1_best_gain
        elif len(gain_ratios[0]) == 0:
            return sorted(gain_ratios[1], key=lambda x: x[0])[-1]
        else:
            return sorted(gain_ratios[0], key=lambda x: x[0])[-1]

    def _calc_entropy(self, values):
        y_col = values[:, -1]
        prob_y_1 =  len(y_col[y_col == 1])/len(y_col)
        prob_y_0 = len(y_col[y_col == 0])/len(y_col)
        if prob_y_0 == 0 or prob_y_1 == 0:
            return 0
        else:
            return -1*((prob_y_1 * math.log2(prob_y_1)) + (prob_y_0 * math.log2(prob_y_0)))

    def predict(self, X):
        if not self.has_trained:
            raise Exception("Train the model before use")  
        
        if not isinstance(X[0], int):
            preds = []
            for x in X:
                preds.append(self.root_node.predict(x))
            return preds
        else:
            return self.root_node.predict(X)

    def print_tree(self):
        self.root_node.print_tree()     
    
class Node:
    def __init__(self, is_root = False):
        # For leaf nodes
        self.pred_value = None
        self.split_var = None
        self.training_points = []

        self.is_root = is_root
        self.candidate_splits = None
        
        # For non-leaf nodes
        self.split_cutoff = None
        self.left_node = None
        self.right_node = None

        self.depth = 0

    def predict(self, X):
        if self.split_cutoff is None:
            return self.pred_value
        elif X[self.split_var] <= self.split_cutoff:
            return self.left_node.predict(X)
        else:
            return self.right_node.predict(X)
        
    def get_prediction(self):
        return self.pred_value
    
    def print_tree(self, level=0, text="root"):
        if self.pred_value is not None:
            print("    " * (level-1), text, f"class: {self.pred_value}")
        else:
            print("  " * level, self.split_cutoff, text, f"on x_{self.split_var}")
        if self.left_node is not None:
            self.left_node.print_tree(level + 1, "left")
        if self.right_node is not None:
            self.right_node.print_tree(level + 1, "right")

    def count_nodes(self):
        if self.pred_value is not None:
            return 1
        else:
            left_nodes, right_nodes = (0,0)
            if self.left_node is not None:
                left_nodes = self.left_node.count_nodes()
            if self.right_node is not None:
                right_nodes = self.right_node.count_nodes()
            return 1 + left_nodes + right_nodes

if __name__ == "__main__":
    data = np.loadtxt("Dbig.txt", dtype=float)

    tree = Tree(data)
    tree.train()
    tree.print_tree()


