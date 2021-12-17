# Do words have (our) characters?

## Data story

https://mismayil.github.io/words-personalities/

### Contribution Details of all group members: 
(with final notebooks mentioned in the brackets)

- Kolic: (ControlGroup_Top1000.ipynb & SuicideAnalysis.ipynb)
  - Preliminary Analysis: control group generation
  - Suicide Analysis: Data cleaning for the analysis, histogram, independent t-test
  - Data Story: Suicide Section

- Mete: (gender.ipynb & country.ipynb)
  - Gender Analysis: Data cleaning for the analysis, heatmap, histograms, independent t-test, validation with external resource
  - Country Analysis: Data cleaning for the analysis, interactive world map
  - Data Story: Introduction, Gender & Country Section

- Rohith: (entrepreneur_artist_compare.ipynb)
  - Entrepreneur & Artist Analysis: Data cleaning for the analysis, heapmap, histograms, validation with external resource
  - Data Story: Entrepreneur & Artist Section

- Yiren: (final_politician.ipynb & app.R & R_for_politician.Rmd)
  - Politician Analysis: Data cleaning for the analysis, heatmap Shiny app, validation with external resource & control group, PCA, K-means Clustering
  - Gender Analysis: Decision Tree and Cross Validation
  - Data Story: Politician Section

## Abstract
Language is the main mode of communication through which a person expresses their thoughts and feelings. This then raises an interesting question: Can language use reflect personality types?  There have been several studies on this topic that have shown significant correlations between different word categories and personality types. In this project, we exploit these correlations to identify personality characteristics for different speakers in the Quotebank dataset. Similarities or differences of these personality styles of groups or individuals are then used to answer some interesting psycho-sociolinguistic questions about people across professions, countries, races and ethnicities among other categories.

## Research questions
Through this project, we are trying to explore a few specific topics given the questions of interest.
1. What are the personality traits of the different famous political leaders of the world and what similarities and differences can we find between them?
2. What are the personality traits of the different billionaires and CEOs? Is there something that makes them different from others?
3. What are the personality traits of different gender, race and ethnicities and have these traits changed over the years?
4. An analysis of change in emotional traits of people affected by an event, such as covid-19.
5. Analysis of personality traits of people with mental disorders like depression, bipolar disorder etc. and suicide victims.


## Proposed additional dataset
We will not be using any additional data other than Quotebank and Wikidata, which were already provided.
 
## Methods
### Personality Analysis
We will be doing a category-based analysis of personality from word usage. We have found several psycho-linguistic research on this topic, especially by Dr. James Pennebaker who is known for his work on revealing the power of pronouns on predicting people's emotions. Most of these papers (see `papers` folder) focus on finding the correlation between personality types and LIWC word categories using supervised learning methods. After fine-grained research through the different papers, we have decided to use the correlation results found between the Big Five personality types (and their subtypes) and LIWC word categories, from a large-scale study of personality and word use among bloggers [1].

All of the analysis that we will be doing is based on personality traits of emotional traits of speakers. So, for each analysis, we have to collect and aggregate all the quotes from a subgroup of speakers for the particular analysis. Since the data that we are handling is huge, our strategy for analysing each event is to extract quotes and the necessary data for the particular analysis. Once the quotes for each speaker is extracted, we will be running it through the academic version of LIWC software [2] to obtain representations of quotes across 66 different word categories. We will be then taking the inner product of this LIWC representation matrix (normalized across categories) and the correlation matrix produced by [1] (see `data/LIWC_OCEAN.csv`) to compute personality scores for each speaker. These scores will be then used along with other speaker attributes to answer some research questions mentioned above.

### Data Loading and Preprocessing
The quote bank data was loaded chunk by chunk to fit in the memory. We are using quotes from 2015 to 2020. As for the wikidata, we are using the parquet file provided for retrieving additional metadata and it is sufficient for the current research questions. However, if we find some interesting analysis in the future and we need extra information for that which is not available in the parquet file, we will be creating our own parquet file by querying the necessary information from https://query.wikidata.org/.

Since quotebank is such a huge dataset we expected it to have some anomalies. As the first step of preprocessing we have decided to take only the quotes that have the probability of 80% or more to be spoken by the speaker. We chose to give more priority to the longer quotes since they more accurately represent a person's speech. We are removing quotes with word length more than 50 as these words usually found to be some garbage string or atypical concatenation of several words that is hard to analyze. We then remove quotes with URLs and JSON-like (key, value) pairs. Then we also filter quotes with containing some percentage (more than 10%) of special characters in them as they are unlikely to represent a real speech.

### Data Analysis
To clearly test if we can analyse and get good results from this data, we did a sample personality analysis on 100 politicians with the most number of quotes in quotebank from both the Democratic and Republican party of USA. For each of these politicians we compile all the quotes and make a string of fixed length. This string is then used to do the personality analysis and get the personality vector. The analysis and the results are shown in detail in the notebook (see analysis.ipynb).

## Proposed timeline
| Date            | Task                                         |
| --------------- | ---------------------------------------------|
| Nov 27 - Dec 4  | Initiation of data analysis task for different research questions |
| Dec 5 - Dec 11  | Analysis result discussion and data story structuring |
| Dec 11 - Dec 17 | Notebook structuring and Data story creation |

## Organization within
As the next step, we have decided to divide the research questions equally among each of us and do the analysis. These analyses will be discussed among the four of us to see the significance of the results. The workload will be divided in a way that a few will be working on the final data story website.

## Questions for TAs
1. Can we use the academic version of LIWC for data analysis?
2. Can the inner product of the word frequencies of the LIWC word categories and the correlation matrix be considered as a good method to find the personality matrix?

## References:

[1] Yarkoni, T. (2010). Personality in 100,000 words: A large-scale analysis of personality and word use among bloggers. Journal of Research in Personality, 44(3), 363–373. https://doi.org/10.1016/j.jrp.2010.04.001 

[2] Pennebaker JW, Francis ME, Booth RJ. Mahway. New Jersey: Lawrence Erlbaum Associates; 2001. Linguistic inquiry and word count: LIWC 2001.
