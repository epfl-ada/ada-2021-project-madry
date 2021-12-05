import re
import datetime
import pandas as pd

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