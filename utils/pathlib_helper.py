import re, os, sys
from itertools import groupby
import json
from pathlib import Path

class FileProcessing():
  
    def __init__(self,log_path, search_path):
        self.log_path = log_path
        self.search_path = search_path

    def replace_double(self):

        search_str = str(Path(self.search_path, '*.json'))
        files = list(self.log_path.rglob(search_str.replace('[', '[[]')))

        search_str = str(Path(self.search_path, '*.pkl'))
        files.extend( list(self.log_path.rglob(search_str.replace('[', '[[]'))) )

        for file in files:
            file_new = re.sub(" \(\d\)", '', str(file)) 
            if str(file) != (file_new):
                try:
                    os.renames(file, file_new) 
                    print(str(file))
                    continue
                except:  ## if failed, remove file
                    os.remove(file)
                    print('File exists, removed')
                    continue
                else:
                    break

    def remove_empty_folders(self, log_path):
        'Function to remove empty folders'

        if not os.path.isdir(log_path):
            return

        # remove empty subfolders
        files = os.listdir(log_path)

        if len(files):
            for f in files:
                fullpath = os.path.join(log_path, f)
                if os.path.isdir(fullpath):
                    self.remove_empty_folders(fullpath)


        # if folder empty, delete it
        files = os.listdir(log_path)
        if len(files) == 0:
            print("Removing empty folder:", log_path)
            os.chmod(log_path, 0o777)
            os.removedirs(log_path)


    def preprocessing_data_loading(self, iter_max=1):
        '''
        rename folders that have the same names
        |-> additional string (1), (2), ... can occur when logs are saved simultaneously from different copies of the notebook
        then delete empty folders
        '''

        self.replace_double()

        i=0
        while i < iter_max:
            i+=1
            try:
                self.remove_empty_folders(self.log_path)
                continue
            except:  ## if failed, report it back to the user ##
                print ("An error occured while trying to remove folder.")
                continue
            else:
                break

    def group_files(self, data_extension='*.json', verbose=True):

        file_parents = []
        search_str = str(Path(self.search_path, data_extension))
        files = list(self.log_path.rglob(search_str.replace('[', '[[]')))
        print(self.log_path)

        for file in files:
            if 'pkl' in data_extension:
              file_parents.append(file.parent.parent)
            else: 
              file_parents.append(file.parent)


        eq_parents = [list(g) for k, g in groupby(file_parents)] #--> AAAA BBB CC D  
        eq_files = [[]]*len(eq_parents)

        for idx, a in enumerate(eq_parents):
            p = a[0].glob(data_extension)
            eq_files[idx] = [x for x in p if x.is_file()]

        if verbose==True:
          string_file_parent = []
          for idx, i in enumerate(eq_parents):
              str_print = str(i[0])
              split_string = str_print.split("logs", 1)
              print('~'*len(split_string[1]))
              print( split_string[1] , '| x ', len(i), ' | Idx: ', idx, ' | ')
              string_file_parent.append( split_string[1] )
          self.string_file_parent = string_file_parent

        self.eq_files = eq_files
        self.eq_parents = eq_parents


    def load_grouped_files(self, index=[0], verbose=True):

        files_plot_nested = [*[self.eq_files[i] for i in index]]
        files_plot_flatten = [item for sublist in files_plot_nested for item in sublist] 

        # Load data
        data = []
        for file in files_plot_flatten:
            with open(file) as json_file:
                json_dict = json.load(json_file)
                data.append(json_dict)
        if verbose:
            print('*'*30,'Data Loaded!', '*'*30)

        return data

