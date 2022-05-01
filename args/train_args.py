import os.path as osp
class Arguments():
    def __init__(self, working_dir):
        self.seed = 1
        self.workers = 2
        self.print_freq = 200
        self.lr = 1e-2
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 4
        self.batch_size = 32
        # k-at-hop: nargs = '+'
        self.k_at_hop = [200, 10]
        self.active_connection = 10
        self.val_feat_path = osp.join(working_dir, './facedata/CASIA.feas.npy')
        self.val_knn_graph_path = osp.join(working_dir, './facedata/knn.graph.CASIA.kdtree.npy')
        self.val_label_path = osp.join(working_dir, './facedata/CASIA.labels.npy')
        self.logs_dir = osp.join(working_dir, 'logs')
