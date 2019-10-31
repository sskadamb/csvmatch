import sys
import dedupe
import colorama

"""
check the relavant documentation here:
https://docs.dedupe.io/en/latest/API-documentation.html#Dedupe.prepare_training
"""

def setup(fields1, fields2,training_json):
    def executor(data1, data2,training_json):
        input1 = {i: {fields1[j]: value for j, value in enumerate(row)} for i, row in enumerate(data1)}
        input2 = {i: {fields1[j]: value for j, value in enumerate(row)} for i, row in enumerate(data2)}
        fields = [{'field': field, 'type': 'String'} for field in fields1]
        #TODO: So RecordLink is the class, and linker is the object that you need to call prepare_training() on.
        linker = dedupe.RecordLink(fields)
        if training_json is None:
            linker.sample(input1, input2, sample_size=1500)
        else:
            linker.prepare_training(input1,input2,training_file=training_json)
        # linker.sample(input1, input2, sample_size=1500) #this takes examples from our data. So yeah... maybe prepare_training() needs to go right before this?
                                                        #seems like prepare_training() already called sample()
        while True:
            labelling(linker)
            try:
                linker.train() #TODO: somewhere here, I need to understand where to place the prepare_training(). probably before this loop, because this loop is where you
                break           #add the manual training examples by hand, so doesn't make any sense.... probably labelling just does the question loop that the user sees. and then
                                #labels from the answer that it gets. That means that... most likely, I put the prepare_training() before this whole thing in between this sample method.
            except ValueError: sys.stderr.write('\nYou need to do more training.\n')
        threshold = linker.threshold(input1, input2, recall_weight=1)
        pairs = linker.match(input1, input2, threshold)
        matches = []
        for pair in pairs:
            matches.append((pair[0][0], pair[0][1], pair[1]))
        return matches
    return executor

def labelling(linker):
    colorama.init()
    sys.stderr.write('\n' + colorama.Style.BRIGHT + colorama.Fore.BLUE + 'Answer questions as follows:\n y - yes\n n - no\n s - skip\n f - finished' + colorama.Style.RESET_ALL + '\n')
    labels = { 'distinct': [], 'match': [] }
    finished = False
    while not finished:
        for pair in linker.uncertainPairs():
            if pair[0] == pair[1]: # if they are exactly the same, presume a match
                labels['match'].append(pair)
                continue
            for record in pair:
                sys.stderr.write('\n')
                for field in set(field.field for field in linker.data_model.primary_fields):
                    sys.stderr.write(colorama.Style.BRIGHT + field + ': ' + colorama.Style.RESET_ALL + record[field] + '\n')
            sys.stderr.write('\n')
            responded = False
            while not responded:
                sys.stderr.write(colorama.Style.BRIGHT + colorama.Fore.BLUE + 'Do these records refer to the same thing? [y/n/s/f]' + colorama.Style.RESET_ALL + ' ')
                response = input()
                responded = True
                if   response == 'y': labels['match'].append(pair)
                elif response == 'n': labels['distinct'].append(pair)
                elif response == 's': continue
                elif response == 'f': finished = True
                else: responded = False
    linker.markPairs(labels)
