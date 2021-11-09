# Project Proposal

**Title:**\
Do words have (our) character?

**Abstract:**\
Speech is the main mode of communication through which a person expresses himself. So, it is natural for each and every person to have his/her own way of using a language to speak. There have been several studies on this topic that have shown significant correlations between different word categories and personality types. This intrigued us to exploit this correlation to identify personality characteristics for different speakers in the quotebank dataset. Similarities or differences of these personality characteristics of groups or individuals will be used to answer some specific questions about the world.

**Research question:**
Through this project, we are trying to explore a few specific topics given the questions of interest.
1. What are the personality traits of the different famous political leaders of the world and can we find a relation between these personality traits and why they are leaders
2. What are the personality traits of the different billionaires and CEOs? Is there something that makes them different from others?
3. Personality traits of different gender, race and ethnicities and has these traits changed over the years?
4. An analysis of change in emotional traits of people affected by an event, such as covid-19.
5. Analysis of suicide victims, depression, bipolar disorder and their personality traits.


**Proposed additional dataset:**\
We will not be using any additional data other than Quotebank and Wikidata, which were already provided.
 
**Methods:**\
*Personality Analysis*\
We will be doing a category-based analysis of personality from word usage. We have found much research on the same. Most of these papers focus on finding the correlation between personality types and LIWC word categories using supervised learning methods. After fine-grained research through the different papers, we have decided to use the correlation matrix between the big five personality types (and their subtypes) and LIWC word categories, from a highly cited paper.

All of the analysis that we will be doing is based on personality traits of emotional traits of speakers. So, for each analysis, we have to collect and aggregate all the quotes from a subgroup of speakers for the particular analysis. Since the data that we are handling is huge, our strategy for analysing each event is to extract quotes and the necessary data for the particular analysis. Once the quotes for each speaker is extracted, we will be running it through the LIWC to categorize the quotes to 70 different word categories and find the word frequency for each category. We will be taking the inner product of the word frequencies and the correlation matrix to get a vector with each value corresponding to each personality type. This vector represents the personality of the speaker.

*Data Handling*\

*Data Preprocessing*\
Since quotebank is such a huge dataset we expected it to have some anomalies. As the first step of preprocessing we have decided to take only the quotes that have the probability of 80% or more to be spoken by the speaker. We chose to give more priority to the longer quotes since they more accurately represent a person's speech. Then we saw that there were 


**Proposed timeline:**
1. Analysis task division and initiation - Nov 27 - Dec 4
2. Discussion and story structuring - Dec 5 - Dec 11
3. Notebook structuring and Data story creation - Dec 11 - Dec 17

**Organization within:**\
As the next step, we have decided to divide the research questions equally among each of us and do the analysis. These analyses will be discussed among the four of us to see the significance of the results. The workload will be divided in a way that a few will be working on the final data story website.

**Questions for TAs:**
1. Can we use the academic version of LIWC for data analy?
2. Can the inner product of the word frequencies of the LIWC word categories and the correlation matrix be considered as a good method to find the personality matrix?
