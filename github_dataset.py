import os
import pathlib
import sys
import torch
from typing import Optional, Tuple

import transformers
from torch.utils.data import Dataset


class GithubDataset(Dataset):
    """
    Torch dataset the provides access to pre-tokenized data stored on the FS
    """

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.AutoTokenizer,
                 file_glob: str = "*.dat",
                 pad_sequences: bool = True,
                 max_seq_len: int = 128,
                 samples_per_file: Optional[int] = None):
        """
        Constructor
            -   In case the samples_per_file is None, the class tries to detect it automatically.
                If the automatic detection fails, a RuntimeError is thrown.
        :param data_path: directory that contains data files directly or subfolders with the data files
        :param tokenizer: tokenizer used for tokenizing the data
        :param file_glob: GLOB used for searching for the data files in given directory
        :param pad_sequences: flag indicating whether to pad the sequences into the max_seq_len or not
        :param max_seq_len: maximum length of the sequences into which the examples shall be padded or trimmed
        :param samples_per_file: how many data points can be found in each dataset file - if None, automatic detection
                                 takes place
        """
        self.__data_path = data_path
        self.__max_seq_len = max_seq_len
        self.__pad_sequences = pad_sequences

        # special tokens indices
        self.__start_token_idx = tokenizer.bos_token_id
        self.__end_token_idx = tokenizer.eos_token_id
        self.__pad_token_idx = tokenizer.pad_token_id

        # file caching information
        self.__last_file_idx = sys.maxsize
        self.__sample_buffer = []

        # get all files in the dataset
        self.__files = [py_file for py_file in pathlib.Path(self.__data_path).rglob(file_glob)]

        # read list of corrupted files and exclude them from the dataset
        with open(os.path.join(self.__data_path, "corrupted.txt"), "r", encoding="utf-8") as corr_f:
            for line in corr_f:
                try:
                    line = line.strip()
                    self.__files.remove(pathlib.Path(line))
                except Exception:
                    # in case the file is not in the list, let's continue
                    print(f"ERROR: Excluded file {line} was not found in the dataset file list!")

        # set number of examples per file or detect it if not provided
        self.__samples_per_files = samples_per_file if samples_per_file is not None \
            else self.__detect_samples_per_file()

    def __getitem__(self, index) -> torch.Tensor:
        """
        Get dataset item on specified index.
        Uses cached dataset file in order to speed up sequential access
        :param index: index of item to be retrieved
        :return: Tensors including (label token ids, decoder input token_ids, encoder input token_ids,
                                    decoder padding mask, encoder padding mask)
        """
        file_idx = index // self.__samples_per_files
        if file_idx != self.__last_file_idx:
            try:
                self.__sample_buffer = torch.load(self.__files[file_idx])
                self.__last_file_idx = file_idx
            except EOFError:
                # if a corrupted file is detected, it shall be excluded from the dataset and shall be written into a
                # file with all corrupted files
                to_remove = self.__files[file_idx]
                print(f"ERROR: Dataset file {to_remove} is corrupted -> skipping...")
                self.__files.remove(to_remove)
                with open(os.path.join(self.__data_path, "corrupted.txt"), "a", encoding="utf-8") as corr_f:
                    corr_f.write(f"{str(to_remove)}\n")
                return self.__getitem__(index)

        example_idx = index % self.__samples_per_files

        # get the example at the given idx
        item = self.__sample_buffer[example_idx]
        target = item.clone()

        # Pad and trim both target, decoder input and the encoder input
        decoder_input, decoder_mask = self.__trim_or_pad(target.clone(), True, False) # decoder input
        target, target_mask = self.__trim_or_pad(target, False, True)   # labels
        item, item_mask = self.__trim_or_pad(item, False, False)        # encoder input

        # replace [PAD] in the labels with -100 for the loss calculation
        pad_positions = (target == self.__pad_token_idx).nonzero().view(-1)
        target[pad_positions] = -100

        return torch.stack([target, decoder_input, item, decoder_mask, item_mask])

    def __trim_or_pad(self, item: torch.Tensor, add_start: bool, add_end: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trim or pad the given dataset item into desired length
        :param item: item to be trimmed or padded
        :param add_start: indicates whether to add [START] token
        :param add_end: indicates whether to add [END] token
        :return: trimmed/padded example
        """
        # define how many tokens would be added to the sequence
        num_added = 0
        num_added = num_added + 1 if add_start else num_added
        num_added = num_added + 1 if add_end else num_added

        mask = torch.ones((self.__max_seq_len,), dtype=torch.int8)

        # trim if necessary
        item_len = item.size()[-1]
        if item_len > self.__max_seq_len - num_added:     # -num added because there should be a space left
            item = item[:self.__max_seq_len - num_added]  # either for the [START] or [END]

        # wrap item with [START] ... [END] and reflect it in the mask based on the settings
        if add_start:
            item = torch.cat((torch.tensor((self.__start_token_idx,)),
                              item
                              ))
            mask[0] = 0  # mask [START]

        if add_end:
            item = torch.cat((item,
                              torch.tensor((self.__end_token_idx,))
                              ))
            mask[item.size()[-1] - 1] = 0  # mask [END]

        # pad sequence
        item_len = item.size()[-1]
        if self.__pad_sequences and item_len < self.__max_seq_len:
            padding = [self.__pad_token_idx] * (self.__max_seq_len - item_len)
            item = torch.cat((
                item,
                torch.tensor(padding)
            ))
            # reflect padding in the mask
            mask[-len(padding):] = 0
        return item, mask

    def __len__(self) -> int:
        """
        Get number of samples in the dataset
        :return: length of the dataset
        """
        return self.__samples_per_files * len(self.__files)

    def __detect_samples_per_file(self) -> int:
        """
        Tries to automatically detect how many samples are in each of the
        dataset files. In case the detection is not successful a RuntimeError exception is thrown.
        :return: number of samples in each of the dataset files
        """
        try:
            test_file = self.__files[0]
            examples = torch.load(test_file)
            return len(examples)
        except Exception as e:
            print(e)
            raise RuntimeError("The GithubDataset was not able to detect the number of samples in each dataset file!")

