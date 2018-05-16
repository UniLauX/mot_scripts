import os

##'===============================MOT devkit parameters setup==========================================='   
motchallenge_devkit_dir='/mnt/phoenix_fastdir/dataset/backup/MOT/motchallenge-devkit'
motchallenge_data_dir=os.path.join(motchallenge_devkit_dir,'data')
seq_map_dir=os.path.join(motchallenge_data_dir,'SeqMaps')
seq_map_ext='.txt'

##general 
upper_mot='MOT'
lower_mot='mot'
full_years=['2015','2016','2017']
short_years=['15','16','17']
set_properties=['train','val','test']


def get_short_year(year_id):
    return short_years[year_id]

def get_full_year(year_id):
    return full_years[year_id]

def get_set_property(set_pro_id):
    return set_properties[set_pro_id]

def get_dataset_name(s_year):
    dataset_name=upper_mot+s_year
    return dataset_name

def get_seqs_data_dir(dataset_name,set_pro):
    seqs_data_dir=os.path.join(motchallenge_data_dir,dataset_name,set_pro)
    return seqs_data_dir

def get_seq_map_name(s_year,set_pro):
    seq_map_name=upper_mot+s_year+'-'+set_pro
    return seq_map_name

def get_seq_map_path(seq_map_name):
    seq_map_path=os.path.join(seq_map_dir,seq_map_name+seq_map_ext)
    return seq_map_path