import pandas as pd
import numpy as np
import sys
from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string

class HashPartitioner:
    def __init__(self, current_partition, num_partitions):
        self.current_partition = current_partition
        self.num_partitions = num_partitions

    def in_current_partition(self, partition_key):
        return self.current_partition == hash(partition_key) % self.num_partitions

class SAX:
    def __init__(self, df_UsageDetails, key_field_name, date_field_name, value_field_name, window_length, select_fields = [], partitioner = HashPartitioner(0, 1), partial_latest_snapshot = False):
        self.window_length = window_length
        self.df_UsageDetails = df_UsageDetails
        self.key_field_name = key_field_name
        self.date_field_name = date_field_name
        self.value_field_name = value_field_name
        self.select_fields = select_fields
        self.partitioner = partitioner
        self.partial_latest_snapshot = partial_latest_snapshot

    def hamming_distance(self, str1, str2, l, modified=True):
        if ((len(str1) < l) or (len(str2)) < l):
            return -1

        distance = 0

        index = 0
        while (index < l):
            if (str1[index] != str2[index]):
                if (modified == True):
                    distance += (1 + index) * abs(ord(str1[index]) - ord(str2[index]))
                else:
                    distance += 1

            index += 1

        return distance

    def get_ranked_discords(self, sequence_string, window_length, descending=True):
        sequence_length = len(sequence_string)
        all_discords = {}

        outer_index = window_length
        while (outer_index < (sequence_length - window_length + 1)):
            inner_index = 0
            min_match_distance = sys.maxsize
            current_sequence = sequence_string[outer_index: (window_length + outer_index)]
            while (inner_index < (sequence_length - window_length + 1)):
                if (abs(outer_index - inner_index) >= window_length):
                    candidate_sequence = sequence_string[inner_index: (window_length + inner_index)]
                    distance = self.hamming_distance(current_sequence, candidate_sequence, window_length)

                    if (distance < min_match_distance):
                        min_match_distance = distance

                inner_index += 1

            if (min_match_distance == sys.maxsize):
                outer_index += 1
                continue

            all_discords[outer_index] = min_match_distance
            outer_index += 1

        last_discord_distance = -1
        ranked_discord_indexes = {}

        ranked_discords = sorted(all_discords.values(), reverse=descending)
        for discord_distance in ranked_discords:
            if discord_distance == last_discord_distance:
                continue
            else:
                last_discord_distance = discord_distance

            for discord_index in all_discords:
                if all_discords[discord_index] == discord_distance:
                    window_length_indexes = []
                    [window_length_indexes.append(index) for index in range(discord_index, discord_index + window_length)]
                    [window_length_indexes.append(index) for index in range(discord_index - window_length + 1, discord_index)]

                    if(len(set(ranked_discord_indexes.keys()) & set(window_length_indexes)) > 0):
                        continue

                    ranked_discord_indexes[discord_index] = discord_distance

        return ranked_discord_indexes

    def get_sequence_string(self, filtered_df):
        numerical_sequence = filtered_df[self.value_field_name].to_numpy()

        sequence_length = len(numerical_sequence)
        if self.partial_latest_snapshot == True:
            sequence_length -= 1

        alpha_representation = ts_to_string(
            znorm(numerical_sequence[0:sequence_length]), cuts_for_asize(4))

        return alpha_representation

    def __iter__(self):
        self.key_grouped_df = self.df_UsageDetails.groupby(by=self.key_field_name).sum().reset_index()
        self.iterator_index = -1
        self.num_rows = self.key_grouped_df.shape[0]

        return self

    def __next__(self):
        discords_for_key = []
        
        while(True):
            self.iterator_index += 1
            if self.iterator_index == self.num_rows:
                raise StopIteration

            row = self.key_grouped_df.iloc[self.iterator_index]
            if self.partitioner.in_current_partition(row[self.key_field_name]) == False:
                continue

            key_filtered_df = self.df_UsageDetails[self.df_UsageDetails[self.key_field_name] == row[self.key_field_name]].sort_values(by=[self.date_field_name])
            sequence_string = self.get_sequence_string(key_filtered_df)
            discords = self.get_ranked_discords(sequence_string, self.window_length)

            discord_rank = 1
            for discord_index in discords.keys():
                discord_distance = discords[discord_index]

                if discord_distance < (self.window_length * 1.5):
                    continue

                output_row = []
                discord_row = key_filtered_df.iloc[discord_index]
                for select_field in self.select_fields:
                    output_row.append(discord_row[select_field])
                
                output_row.append(discord_row[self.date_field_name])
                output_row.append(discord_rank)
                output_row.append(discord_row[self.value_field_name])

                discords_for_key.append(output_row)
                discord_rank += 1
            
            if len(discords_for_key) > 0:
                return discords_for_key