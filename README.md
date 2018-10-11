# sys6018-competition-blogger-characteristics
Private GitHub Repository for Competition #3 on Kaggle
Charu Rawat, Bowei Sun, & Ashwanth Samuel

REFLECTION
  Setup:
This competition involved analyzing an array of blogger entries, including their associated features such as gender, topic (occupation,) sign and date of posting, in order to predict the age of the blogger. Most bloggers, recognized by their user.id, posted more than once, thus we have duplicate data points for each unique user.id. Initially, null values and missing data was cleaned and removed from the dataset. Then, a sentiment analysis was utilized in order to associate the words within the blogger entries with a specific score (through a tf-idf methodology,) and this was then used in conjunciton with the aforementioned variables (of which some had to experience one-hot encoding in order to be sustainabily utilized) in order to develop the optimal linear regression model to predict the ages of the bloggers. 
  
  Who Might Care about this Problem and Why?:
This issue of using different types of data to predict the age of a user is definitely scalable and usable for those who might care about the questions that it answers, as well as the questions it raises. For example, the hosting site of the blog posts may find it very important to understand the age of its users in order to optimize user experience; If a user is below a certain age threshold, it may be important that they are only seeing censored sites and entries. On the other hand, the site may be able to target users of a certain age with other data entries that the user will enjoy perusing. Thus, the user could potentially enjoy their experience much more and continue to visit the site. 

In a similar vein, the user data could be sold to advertising agencies and these firms could target specific age groups with relevant advertising material with the hopes that they will potentially appeal to these specific age groups. Advertising agencies frequently utilize this type of thinking in their preference algorithms, so if they were able to have access to all of this data from the site, they could increase the overall effectiveness of their advertising techniques. 

Furthermore, other sites that allow for user reviews or comments can utilize this data as well. Amazon, for instance, could take all of the reviews from their site and understand the age of each reviewer. From that data, they could then provide personalized recommendations for other products that people of similar ages may buy. 

  What Made this Problem Challenging?:
This competition was very difficult for a couple of reasons. First, was the sheer size of the dataset that the team was manipulating. It took about two minutes to even read in the data from a csv file, so one can imagine the time it took to operate that data and its corresponding functions. Second, our team decided to use R because of our increased familiarity with the software as well as its relevant references that were brought up during class. We found, however, that R is not the most ideal software to use for an analysis of this nature because of its very slow processing time. Third, we found that we understood the basic algorithm behind the tf-idf matrix and its relevant methodology, but implementing it on this large of a scale proved to be incredibly difficult. 

Ultimately, none of us had ever worked with text before and this proved to be quite challenging. Even after splitting sentences, removing grammar, and finding frequencies of words, there were still so many types of words that it was difficult to choose the ones that were the most important. Even further, many of the factors that we unearthed through the text mining process, just really did not seem to correlate with age. We can understand how in certain scenarios, the actual content of the blogs will affect age, but in this case, they seemed to not. 

  What other Problems Resemble this Problem?
This type of problem can be resembled in many other scenarios as well. For example, a blogging site could utilize this analysis to guess someone’s gender, education level, wealth status, and a whole other gamut of factors that the site may find interesting. Additionally, this type of analysis can be used retroactively to identify authors with certain texts or vice versa. For example, there was recently a study released analyzing whether John Lennon or Paul McCarthy wrote the song “In My Life.” The data scientists essentially went through all of the songs written by each of the individuals, tagged them with certain characteristics and compared them to the actual song to determine who had written IML. This kind of analysis utilizes the basic facets of what we accomplished here within the scope of text mining.

  Team Roles:
Our team learned a lot while working together on this Kaggle competition. While none of us have extensive backgrounds in text-mining, we spent time to deliver a product representative of our diligent efforts. The steps of this project included collaborating and brainstorming an initial model, taking some time to think about what we needed for our following submissions, understanding the algorithm behind the tf-idf approach, and implementing the steps needed to solve the problem. This proved to be a challenging, yet intriguing task for us and we all worked with our individual strengths to produce our deliverable. The roles were discussed before the start of the project, and were subsequently altered as our thinking and understanding of the problem changed.

Charu was in charge of initially implementing the algorithm and was the main coder of the group. She figured out how to get from the frequency counts of the words to the tf-idf matrix, and subsequently use the appropriate words needed for the feature engineering of the model.

Bowei was tasked with refining the model using his own perspective on the model selection. He was able to select and choose the variables and words that he thought were most informative in relation to predicting the final age of the user.

Ashwanth was the administrative lead on the project, pulling together the submission and taking care of the paperwork. Additionally, he made a few models utilizing feature engineering and building off the initial work that Charu and Bowei completed.

