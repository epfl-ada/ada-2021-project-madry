import re
import datetime
import pandas as pd
from typing import Callable
import bz2
import json
import os
import bz2
import time
import csv
from tqdm import tqdm
from collections import defaultdict
from wikidata.client import Client
import math
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

wiki_client = Client()

PATTERN_INPUT = 'data/quotebank/quotes-{}.json.bz2'
CHUNK_SIZE = 1_048_576

LIWC_OCEAN_MAP = {
    'Pronoun': 'pronouns', 
    'I': 'first_person_sing',
    'We': 'first_person_pl',
    'Self': 'first_person',
    'You': 'second_person',
    'Other': 'third_person',
    'Negate': 'negation',
    'Assent': 'assent',
    'Article': 'articles',
    'Preps': 'prepositions',
    'Number': 'numbers',
    'Affect': 'affect',
    'Posemo': 'positive_emotion',
    'Posfeel': 'positive_feeling',
    'Optim': 'optimism',
    'Negemo': 'negative_emotion',
    'Anx': 'anxiety_fear',
    'Anger': 'anger',
    'Sad': 'sadness',
    'Cogmech': 'cognitive_proc',
    'Cause': 'causation',
    'Insight': 'insight',
    'Discrep': 'discrepancy',
    'Inhib': 'inhibition',
    'Tentat': 'tentative',
    'Certain': 'certainty',
    'Senses': 'sensory_proc',
    'See': 'seeing',
    'Hear': 'hearing',
    'Feel': 'feeling',
    'Social': 'social_proc',
    'Comm': 'communication',
    'Othref': 'other_refs',
    'Friends': 'friends',
    'Family': 'family',
    'Humans': 'humans',
    'Time': 'time',
    'Past': 'past_tense',
    'Present': 'present_tense',
    'Future': 'future_tense',
    'Space': 'space',
    'Up': 'up',
    'Down': 'down',
    'Incl': 'inclusive',
    'Excl': 'exclusive',
    'Motion': 'motion',
    'Occup': 'occupation',
    'School': 'school',
    'Job': 'job',
    'Achieve': 'achievement',
    'Leisure': 'leisure',
    'Home': 'home',
    'Sports': 'sports',
    'TV': 'tv_movies',
    'Music': 'music',
    'Money': 'money',
    'Metaph': 'metaphysical',
    'Relig': 'religion',
    'Death': 'death',
    'Physcal': 'physical_states',
    'Body': 'body_states',
    'Sexual': 'sexuality',
    'Eating': 'eat_drink',
    'Sleep': 'sleeping',
    'Groom': 'grooming',
    'Swear': 'swear_words'
}

PERSONALITY_ATTRS = ['neuroticism', 'anxiety', 'hostility', 'depression',
                     'self_consciousness', 'immoderation', 'vulnerability', 'extraversion',
                     'friendliness', 'gregariousness', 'assertiveness', 'activity_level',
                     'excitement_seeking', 'cheerfulness', 'openness', 'imagination',
                     'artistic_interests', 'emotionality', 'adventurousness', 'intellect',
                     'liberalism', 'agreeableness', 'trust', 'morality', 'altruism',
                     'cooperation', 'modesty', 'sympathy', 'conscientiousness',
                     'self_efficacy', 'orderliness', 'dutifulness', 'achievement_striving',
                     'self_discipline', 'cautiousness']

def predict_personality(liwc_data: pd.DataFrame, sig_level: int = 1) -> pd.DataFrame:
    """Predicts personality based on the LIWC metrics.
    This function computes personality scores based on the LIWC features. It essentially multiplies the matrix of normalized LIWC
    features with the matrix of correlations between LIWC and Big-Five personality types. 
    
    More specifically, here liwc_data is of dimension (N, D) where N is the number of samples (e.g. quotes for speakers) and
    D is the number LIWC features/categories (e.g. first_person_pronoun, negation etc. full list of categories and their descriptions
    can be found in the LIWC manual https://www.researchgate.net/publication/228650445_The_Development_and_Psychometric_Properties_of_LIWC2007)
    The correlations matrix on the other hand is of dimensions (D, K) where K is the number of extended BigFive personality categories 
    (e.g. neuroticism, depression, friendliness etc.) The significance level of these correlations can be also customized using the
    sig_level parameter which ranges between 0 and 3 where 0 means use all the correlations, 1 means use only those with p < 0.05,
    2 means use those with p < 0.01 and 3 means use those with p < 0.001.

    Args:
        liwc_data (pd.DataFrame): LIWC metrics data
        sig_level (int, optional): Significance level. Defaults to 1 (i.e. p < 0.05)

    Returns:
        pd.DataFrame: Personality scores
    """
    # Load LIWC-OCEAN correlations matrix
    liwc_ocean_data = pd.read_csv('data/LIWC_OCEAN.csv', index_col=0)

    # Load significancy level matrix for LIWC-OCEAN (of the same dimension as the correlations matrix)
    liwc_ocean_sig_data = pd.read_csv('data/LIWC_OCEAN_Significance.csv', index_col=0)

    # Rename LIWC features data columns to match correlations index names
    liwc_data = liwc_data[list(LIWC_OCEAN_MAP.keys())].rename(columns=LIWC_OCEAN_MAP)

    # Normalize LIWC features to be between 0 and 1
    liwc_data = liwc_data.div(liwc_data.sum(axis=1), axis=0)

    # Verify that the column names and index names match for matrix multiplication
    assert (liwc_ocean_data.index == liwc_data.columns).all()

    # Filter correlations by significance level
    liwc_ocean_data_with_sig = liwc_ocean_data * (liwc_ocean_sig_data >= sig_level).astype(int)

    # Compute personality scores
    return liwc_data.dot(liwc_ocean_data_with_sig)


def to_datetime(datetime_str):
    matches = re.search('.*(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2}).*', datetime_str)
    year, month, day = int(matches.group('year')), int(matches.group('month')), int(matches.group('day'))
    return datetime.date(max(1, year), min(max(1, month), 12), min(max(1, day), 28))


def read_json(json_file):
    with open(json_file) as f:
        return json.load(f)

def write_json_to_file(name, obj):
    # Use current timestamp to make the name of the file unique
    millis = round(time.time() * 1000)
    name = f'{name}_{millis}.json'
    with open(name, 'wb') as f:
        output = json.dumps(obj)
        f.write(output.encode('utf-8'))
    return name


def check_if_significant_quote(row: dict, significant_quote_counters: dict) -> None:
    """Check if the quote for a given row can be considered significant for analysis and update
    the significancy dictionary.

    Args:
        row (dict): Row of data
        significant_quote_counters (dict): Dict to keep track of significant quotes
    """
    probabilities = row['probas']
    qids = row['qids']
    
    # Check if the probas and qids values exist
    if (len(probabilities) == 0 or len(qids) == 0):
        return
    
    # Check if the speaker is not 'Unknown'
    if (probabilities[0][0] == 'None'):
        return
    
    # Check if the probability is over 80%
    prob = float(probabilities[0][1])
    if (prob < 0.8):
        return
    
    # Increment count
    qid = qids[0]
    significant_quote_counters[qid] = significant_quote_counters.get(qid, 0) + 1


def process_compressed_json_file(input_file_name: str, output_name: str, year: int, process_json_object: Callable) -> str:
    """
    Read from a compressed file chunk by chunk. Decompress every chunk and try to decode it and parse it into an array of JSON objects.
    For each JSON object extracted this way, run the process_json_object function.
    In the end, a JSON object representing the result of this process is written into a file.

    Args:
        input_file_name (str): Name of the compressed json file which is the subject of processing.
        output_name (str): First part of the output file name. Used in creation of the full output file name: the year parameter and a timestamp are appended, as well as the .json extension.
        year (int): Represents the year for which the data in the input file is gathered, is appended to the output_name to generate the full output file name.
        process_json_object (Callable): Function that processes the individual JSON objects extracted from the compressed file. The signature should be as follows:
            Args:
                json_obj: JSON object which is to be processed.
                out_json: The output object in which the result of the processing is stored

    Returns:
        (str) Full name of the output JSON file.
    """
    # Decompression variables
    decompressor = bz2.BZ2Decompressor()
    
    # Decoding variables
    decoding_buffer = bytearray([])
    decoding_error_counter = 0
    
    # Parsing variables
    parsing_buffer = ''
    parsing_error_counter = 0
    
    # Progress variables - used to provide feedback to the dev
    input_size = os.path.getsize(input_file_name)
    start_time = time.time()
    total_in = 0
    total_out = 0
    previous_value = -1
    
    # Result of processing
    out_json = dict()
    
    # Iterate through the file
    with open(input_file_name, 'rb') as input_file:
        for chunk in tqdm(iter(lambda: input_file.read(CHUNK_SIZE), b''), desc=f'Processing year {year}'):
            # Feed chunk to decompressor
            decompressed_chunk = decompressor.decompress(chunk)
            dec_chunk_length = len(decompressed_chunk)
            
            # Check the length of the decompressed data - 0 is common -- waiting for a bzip2 block
            if (dec_chunk_length == 0):
                continue
            
            # Try to decode byte array
            decoding_buffer += decompressed_chunk
            try:
                chunk_string = decoding_buffer.decode('utf-8')
                
                # Clear buffer
                decoding_buffer = bytearray([])
                
                decoding_successful = True
            except UnicodeDecodeError:
                # Error occurs when input stream is split in the middle of a character which is encoded with multiple bytes
                decoding_error_counter += 1
                decoding_successful = False
            
            # Try to parse the decoded string
            if decoding_successful:
                # Elements of the JSON array are split by '\n'
                array_elements = chunk_string.split('\n')
                
                # Iterate through the JSON array in the current chunk
                for json_candidate in array_elements:
                    # Try to parse the JSON object, might fail if the object was divided in parts because of the chunk separation
                    parsing_buffer += json_candidate
                    try:
                        json_obj = json.loads(parsing_buffer)
                        
                        # Clear buffer
                        parsing_buffer = ''
                        
                        parsing_successful = True
                    except ValueError:
                        """
                        Error occurs when the line does not contain the whole JSON object, which happens for the last array element in almost every chunk of input stream.
                        We solve this by remembering the prevous partial objects in parsing_buffer, and then merging it with the rest of the object when we load the next chunk.
                        """
                        parsing_error_counter += 1
                        parsing_successful = False
                    
                    # Perform JSON object processing
                    if parsing_successful:
                        process_json_object(json_obj, out_json)
            
            # Show progress
            total_in += len(chunk)
            total_out += dec_chunk_length
            if dec_chunk_length != 0:    # only if a bzip2 block emitted
                processed_fraction = round(1000 * total_in / input_size)
                if processed_fraction != previous_value:
                    left = (input_size / total_in - 1) * (time.time() - start_time)
                    # print(f'\r{processed_fraction / 10:.1f}% (~{left:.1f}s left)\tyear: {year}\tnumber of entries: {len(out_json)}\tdecoding errors: {decoding_error_counter}\tparsing errors: {parsing_error_counter}', end='      ')
                    previous_value = processed_fraction
    
    # Save result to file
    output_full_name = write_json_to_file(f'{output_name}-{year}', out_json)
    
    # Report ending
    print()
    total_time = time.time() - start_time
    print(f'File {input_file_name} processed in {total_time:.1f}s', end='\n\n')
    
    return output_full_name


def check_if_speaker_quote(row: dict, quotes: dict, speakers: list) -> None:
    """CHeck if speaker quote is useful for analysis

    Args:
        row (dict): Row of data
        quotes (dict): Dict to keep track of quotes
        speakers (list): Speaker QID list
    """
    probabilities = row['probas']
    qids = row['qids']
    
    # Check if the probas and qids values exist
    if (len(probabilities) == 0 or len(qids) == 0):
        return
    
    # Check if the speaker is not 'Unknown'
    if (probabilities[0][0] == 'None'):
        return
    
    # Check if the probability is over 80%
    p = float(probabilities[0][1])
    if (p < 0.8):
        return
    
    # Check if the speaker is on the speaker list
    qid = qids[0]
    if qid not in speakers:
        return
    
    # Remember only the quote and the probability
    data = {}
    data['quotation'] = row['quotation']
    data['proba'] = row['probas'][0][1]
    
    # Append the quote
    arr = quotes.get(qid, [])
    arr.append(data)
    quotes[qid] = arr


def combine_quotes_files(quote_files):
    combined_quotes = defaultdict(list)

    for quote_file in quote_files:
        with open(quote_file) as qf:
            quotes = json.load(qf)

            for qid, quote_lst in quotes.items():
                combined_quotes[qid] += quote_lst

    return combined_quotes


def filter_quotes(quotes):
    filtered_quotes = []

    weird_pattern = '[_@#+&;:\(\)\{\}\[\]\\/`]'
    json_pattern = '\{.*[a-zA-Z]+:\s[\'"`][a-zA-Z0-9]+[\'"`].*\}'
    url_pattern = 'https?'

    for qid, quote_lst in quotes.items():        
        clean_quotes = []
        for entry in quote_lst:
            text = entry['quotation']
            
            longest = max(entry['quotation'].split(), key=len)
            if (len(longest) > 50):
                filtered_quotes.append(entry)
                continue
            
            if re.search(url_pattern, text) is not None:
                filtered_quotes.append(entry)
                continue
            
            if re.search(json_pattern, text) is not None:
                filtered_quotes.append(entry)
                continue
                
            weird_num = len(re.findall(weird_pattern, text))
            total = len(text)
            weird_percent = weird_num / total
            if (weird_percent > 0.1):
                filtered_quotes.append(entry)
                continue
                
            clean_quotes.append(entry)
        quotes[qid] = clean_quotes

    return quotes, filtered_quotes


def concatenate_quotes(quotes, quote_length=5000):
    for qid, quote_lst in quotes.items():
        # Sort the quotes by length
        quote_lst.sort(key = lambda x: len(x['quotation']), reverse = True)
        
        concat = ''
        for quote in quote_lst:
            # Concatenate the quotes
            concat += ' ' + quote['quotation']
            
            # Trim if we are over QUOTE_LENGTH
            if (len(concat) >= quote_length):
                concat = concat[0:quote_length]
                break
        
        quotes[qid] = concat

    return quotes


def write_quotes_to_csv(quotes, output_file):
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['qid', 'quote'])
        for qid, quote in quotes.items():
            writer.writerow([qid, quote])


def get_wiki_entity(qid):
    entity = wiki_client.get(qid, load=True)
    return entity


def get_countries(qids):
    countries = {}
    entity_instanceof = get_wiki_entity('P31')
    entity_country = get_wiki_entity('Q6256')

    for qid in qids:
        entity = get_wiki_entity(qid)
        for instance in entity.getlist(entity_instanceof):
            if instance == entity_country:
                countries[qid] = entity
                break
    
    return countries


def get_iso_code(qid):
    entity_iso = get_wiki_entity('P298')
    entity = get_wiki_entity(qid)
    iso = entity.getlist(entity_iso)
    return iso[0] if len(iso) > 0 else None


def get_country_info(qid):
    entity = get_wiki_entity(qid)
    iso = get_iso_code(qid)
    return {'name': str(entity.label), 'iso': str(iso)}

def number_of_clusters(labels):
    return len(np.unique(labels))

def clustering_percentage_score(labels):
    score = 0
    for val in np.unique(labels):
        number_of_vals = len(labels[labels == val])
        if number_of_vals > score:
            score = number_of_vals
    return (len(labels) - score)

def search_for_epsilon(
    df, 
    cluster_columns, 
    iterations=20,
    clustering_score_function=number_of_clusters
):
    print('Searching for best EPSILON for DBSCAN!')
    lower_eps = 1
    lower_score = clustering_score_function(np.array([lower_eps]))
            
    upper_eps = 20 
    upper_score = clustering_score_function(np.array([upper_eps]))

    best_eps = lower_eps
    best_score = lower_score

    for i in range(iterations):
        mid_eps = (lower_eps + upper_eps) / 2.0
        mid_clusters = DBSCAN(eps=mid_eps).fit(df[PERSONALITY_ATTRS])
        mid_score = clustering_score_function(mid_clusters.labels_)
        print(f'{"eps:":5}{mid_eps:5}\t score: {mid_score}')

        if mid_score > best_score:
            best_eps = mid_eps
            best_score = mid_score
            
        mid_plus = mid_eps + (upper_eps - mid_eps) * 0.1
        mid_minus = mid_eps - (mid_eps - lower_eps) * 0.1

        mid_plus_test = DBSCAN(eps=mid_plus).fit(df[PERSONALITY_ATTRS])
        mid_minus_test = DBSCAN(eps=mid_minus).fit(df[PERSONALITY_ATTRS])
        
        mid_plus_score = clustering_score_function(mid_plus_test.labels_)
        mid_minus_score = clustering_score_function(mid_minus_test.labels_)

        if mid_plus_score > mid_minus_score:
            lower_eps = mid_eps
            lower_score = mid_score
        elif mid_minus_score:
            upper_eps = mid_eps
            upper_score = mid_score
        else:
            if upper_score > lower_score:
                lower_eps = mid_eps
                lower_score = mid_score
            else:
                upper_eps = mid_eps
                upper_score = mid_score
    
    print(f'Best EPSILON:\t{best_eps}')
    return best_eps
            
def scatter_plot_clusters(
    df, 
    cluster_columns,
    hover_name,
    hover_data,
    eps=0, 
    normalize=True,
    clustering_score_function=number_of_clusters
):
    """
    Perform clustering using DBSCAN.
    Visualize the clusters with a scatter plot, using TSNE to reduce dimensionality of data.

    Args:
        df (DataFrame): Clustering is performed using the data from this dataframe.
        cluster_columns (array of str): Names of dataframe columns which are used for clustering.
        hover_name (str): When a scatter point is hovered, this column from the relevant dataframe row is used as a title for the popup card.
        hover_data (array of str): When a scatter point is hovered, these columns from the relevant dataframe row are used as content for the popup card.
        eps (float): Epsilon values used for DBSCAN algorithm. If 0 is passed search_for_epsilon function is invoked in order to get a valid value.
        normalize (bool): Whether to normalize the columns used for clustering.
        clustering_score_function (function): Function used to score how good of a fit a certain value of epsilon is. A higher score is a better score.
            Args:
                labels (array of int): Array of cluster labels. Each element is the cluster label of the relevant dataframe row.
                
            Returns:
                score (int): How good the assigned labels are according to some parameter. Higher is better.
    """
    # Make copy so no side effects are felt outside function
    df = df.copy()
    
    # Normalize columns used for clustering
    if normalize:
        normalized = df[cluster_columns]
        df[cluster_columns]
    
    # Search for best epsilon if not provided
    if eps == 0:
        eps = search_for_epsilon(
            df, 
            cluster_columns, 
            clustering_score_function=clustering_score_function
        )
        
    # Get cluster labels using DBSCAN
    cluster_info = DBSCAN(eps=eps).fit(df[cluster_columns])
    
    # Get 2D representation using TSNE - used just for visualisation
    coordinates = TSNE().fit_transform(df[cluster_columns])
    
    # Add resulting labels and coordinates to DataFrame
    CLUSTER_LABELS_COLUMN_NAME = '__cluster__clusterLabel'
    CLUSTER_X_COORDS_COLUMN_NAME = '__cluster__coordinate_x'
    CLUSTER_Y_COORDS_COLUMN_NAME = '__cluster__coordinate_y'
    
    df[CLUSTER_LABELS_COLUMN_NAME] = cluster_info.labels_
    df[CLUSTER_X_COORDS_COLUMN_NAME] = coordinates[:, 0]
    df[CLUSTER_Y_COORDS_COLUMN_NAME] = coordinates[:, 1]
    
    # Draw scatter plot
    fig = px.scatter(
        df, 
        x=CLUSTER_X_COORDS_COLUMN_NAME, 
        y=CLUSTER_Y_COORDS_COLUMN_NAME, 
        hover_name=hover_name, 
        hover_data=hover_data,
        color=CLUSTER_LABELS_COLUMN_NAME, 
        size_max=60
    )
    fig.update_layout(
         height=800)
    fig.show()


def visualize_world(df, location_col, color_col, hover_cols, title=None, labels=None, animation_col=None, color_scale=px.colors.sequential.Rainbow, range_col=[0, 1]):
    fig = px.choropleth(df, locations=location_col,
                        color=color_col,
                        hover_data=hover_cols,
                        color_continuous_scale=color_scale,
                        labels=labels,
                        title=title,
                        animation_frame=animation_col,
                        range_color=range_col)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_geos(visible=False)
    return fig