# Configurations for the train/test/evaluate phases of trt RNN
# Modify the below parameters for different configurations

isTrain = True #if you are running only test, set this to False

model_name = "MC_new_training" #give a defining name to your model

comment = "New training run with configs..."

data_year = '2016' 

# All of the features available to the RNN - not the ones that are used: define them below
features = ["avgmu", "weight", "trkOcc", "p", "pT", "lep_pT", "eta", "phi", "eProbHT", "nTRThits", 
            "nTRThitsMan", "nTRTouts", "nArhits", "nXehits", "nHThitsMan", "nPrechitsMan",
            "NhitsdEdx", "sumToT", "sumL", "sumToTsumL", "fAr", "fHTMB", "PHF",
            "hit_HTMB", "hit_gasType", "hit_drifttime", "hit_tot", "hit_T0", "hit_L",
            "hit_bec", "hit_rTrkWire", "hit_layer", "hit_strawlayer", "hit_strawnumber",
            "hit_localTheta", "hit_localPhi", "hit_HitZ", "hit_HitR","hit_ZR"]


# All of the track features listed here for reference
#track_features = [b"avgmu", b"weight", b"trkOcc", b"p", b"pT", b"lep_pT", b"eta", b"phi", b"nTRThits", 
#                  b"nTRThitsMan", b"nTRTouts", b"nArhits", b"nXehits", b"nHThitsMan", b"nPrechitsMan",
#                  b"sumToT", b"sumL", b"sumToTsumL", b"fAr", b"fHTMB", b"PHF"]

# Track features to be used in RNN
track_features = [b"trkOcc", b"p", b"pT", b"nXehits", b"fAr", b"fHTMB", b"PHF"]

# All of the hit features listed here for reference
#hit_features = [b"hit_HTMB", b"hit_gasType", b"hit_drifttime", b"hit_tot", b"hit_T0", b"hit_L",
#                b"hit_bec", b"hit_rTrkWire", b"hit_layer", b"hit_strawlayer", b"hit_strawnumber",
#                b"hit_localTheta", b"hit_localPhi", b"hit_HitZ", b"hit_HitR", b"hit_ZR"]

# Hit features to be used in RNN
hit_features = [b"hit_HTMB", b"hit_gasType", b"hit_tot", b"hit_L", b"hit_rTrkWire",
                b"hit_HitZ", b"hit_HitR"]

# Arguments-hyper paremeters
class args(object):
    def __init__(self):
        self.batch_size = 64
        self.epochs = 2 # number of epochs for training
        self.lr = 0.0001 # learning rate
        self.enable_cuda = True
        self.seed = 999
        self.log_interval = 30 # Loss report interval during training - every log_interval batches
        self.statesave = False
        self.train_size = 15000 # size of MC training set
        self.val_size = 1000 # size of MC validation set
        self.test_size = 1000 # size of MC test set
        self.data_test_size = 1000 # size of DATA test set
        self.prob = 0.5 #balance of the electrons and muons in a given batch - 0.5: equal # of e and mu
        self.dense_dim = 128 #number of units in the dense layer
        self.hidden_dim = 64 #number of LSTM units in LSTM layer
        self.show_plots = False #show plots interactively 
        self.report = True #show print statements

#path of the training and testing data for MC and DATA or a given year
data_year = "2016"
data_test_dirs  = ["../data/data_test"]

mc_e_train_dirs = ["../data/e_train_files"]
mc_e_test_dirs  = ["../data/e_test_files"]
mc_m_train_dirs = ["../data/m_train_files"]
mc_m_test_dirs  = ["../data/m_test_files"]

file_paths = {"mc_e_train_dirs": mc_e_train_dirs, "mc_e_test_dirs": mc_e_test_dirs, "mc_m_train_dirs": mc_m_train_dirs, "mc_m_test_dirs": mc_m_test_dirs, "data_test_dirs": data_test_dirs}