# Project Proposal

**Title:**\
Do words have (our) character?

**Abstract:**\
We live in a complex world. We see politicians with very little experience come to power, people who were born in mere poverty become billionaires, global calamities and events like covid affect human life,
and many other events happening around us that affect us directly or indirectly. It is nearly impossible to clearly understand how and why different events are happening around us. 
Through this project we are hoping to answer a few interesting questions that we had about such events that happened in our life. We will be answering these questions by finding a correlation between the event and personality/emotion of people who were involved or were affected by these events.

**Research question:**\
What can personality analysis based on text data tell us about different events in our world?

**Proposed additional dataset:**\
In addition to the quotebank dataset, we will be using wikidata knowledge base for some parts of the analysis. To give an example of how we are gonna use the Wikidata dump,
for some particular analysis, we need extra information like political affiliation, gender, ethnicity, country of origin etc. So, we will be extracting these specific information from the wikidata dump by making some changes to the sample code provided. 
 - Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect.
 - Discuss data size and format if relevant.
 - It is your responsibility to check that what you propose is feasible.
 
**Methods:**\
We will be doing a category based analysis of personality from word usage. We have found much research on the same. Most of these papers focus on finding the correlation between personality types
and LIWC word categories using supervised learning methods. After thorough research through the different papers, we have decided to use the correlation matrix between big five personality types (and their sub types) and LIWC word categories, from a well used paper.

All of the analysis that we will be doing is based on personality traits of emotional traits of speakers. So, for each analysis, we have to collect and aggregate all the quotes from a sub group of speakers
for the particular analysis. Since the data that we are handling is huge, our strategy for analysing each event is to extract quotes and the necessary data for the particular analysis. Once the quotes for each speaker is extracted, 
we will be running it through the LIWC to categorize the quotes to 70 different word categories and find the word frequency for each category. We will be taking the inner product of the word frequencies and the correlation matrix 
to get a vector with each value corresponding to each personality type. This vector represents the personality of the speaker.

We will be using the personality vector of the speakers thus obtained to critically analyse and understand different events that happened throughout the world and the outcomes of the events.

1. What are the different analyses we are gonna do?
2. What are the personality traits of the different famous political leaders of the world and can we find a relation between these personality traits and why they are leaders
3. What are the personality traits of the different billionaires and CEOs? Is there something that makes them different from others?
4. Personality traits of different gender, race and ethnicities and has these traits changed over the years?
5. An analysis of change in emotional traits of people affected by an event.
6. Analysis of suicide victims and their personality traits.

**Proposed timeline:**
1. Discussing the feasibility of different possible project ideas - Oct 25 to Oct 31
2. Data loading and preprocessing - Nov 1 to Nov 7
3. Basic analysis and method checking - Nov 8 to Nov 12
4. Analysis task division and initiation - 
5. Final data compilation and report creation - 

**Organization within:**\
(A timeline of internal milestones up until milestone 3)

**Questions for TAs:**
1. Can we use the academic version of LIWC for data analy?
2. Can the inner product of the word frequencies of the LIWC word categories and the correlation matrix be considered as a good method to find the personality matrix?
