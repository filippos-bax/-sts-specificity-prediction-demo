import yaml

#Function that loads the config file
def load_config(path="config/config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg

#Iterator function that reads a FASTA file and yields each record
def read_fasta_records(file_path):
    with open(file_path, 'r') as file:
        record = ''
        for line in file:
            if line.startswith('>') and record:
                yield record
                record = ('|'.join(line.split('|')[:-1])).strip() + '\n'
            else:
                record += line
        if record:
            yield record
