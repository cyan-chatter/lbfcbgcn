import os.path as osp
class Arguments():
    def __init__(self, working_dir):
        self.seed = 1
        self.workers = 8
        self.print_freq = 40
        self.lr = 1e-5
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 20
        self.batch_size = 32
        # k-at-hop: nargs = '+'
        self.k_at_hop = [20, 5]
        self.active_connection = 5
        self.val_feat_path = osp.join(working_dir, './facedata/1024.fea.npy')
        self.val_knn_graph_path = osp.join(working_dir, './facedata/knn.graph.1024.bf.npy')
        self.val_label_path = osp.join(working_dir, './facedata/1024.labels.npy')
        self.checkpoint = './logs/logs/best.ckpt'