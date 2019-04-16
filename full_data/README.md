# Full Data Description

### This is the full dataset, including both holdout samples (dev/test) for the 2019 SIOP Machine Learning Competition.

It includes the following columns

## Respondent ID
This is a unique identifier for each participant

## 5 open-ended responses

Each open-ended prompt was designed to elicit a specific personality trait.

#### Agreeableness
1. A colleague of yours has requested vacation for the same week as you. According to your supervisor one of you has to take a different week of vacation because it would be too busy at work if both of you are absent. Your colleague is not willing to change their vacation plans. What would you do and why? 
#### Conscientiousness
2. You have a project due in two weeks. Your workload is light leading up to the due date. You have confidence in your ability to handle the project, but are aware sometimes your boss gives you last tasks that can take significant amounts of time and attention. How would you handle this project and why?
#### Extraversion
3. You and a colleague have had a long day at work and you just find out you have been invited to a networking meeting with one of your largest clients. Your colleague is leaning towards not going and if they don't go you won’t know anyone there. What would you do and why? 
#### Neuroticism
4. Your manager just gave you some negative feedback at work. You don’t agree with the feedback and don’t believe that it is true. Yet the feedback could carry real consequences (e.g., losing your annual bonus). How do you feel about this situation? What would you do? 
#### Openness
5. The company closed a deal with a client from Norway and asks who would like to volunteer to be involved on the project. That person would have to learn some things about the country and culture but doesn't necessarily need to travel. Would you find this experience enjoyable or boring? Why?


## 5 trait scores

Each of the Big 5 trait scores is represented. The data were collected from the [BFI-2 Personality Inventory](http://www.colby.edu/psych/wp-content/uploads/sites/50/2013/08/bfi2-form.pdf). 

- E_Scale_score: the average response on a 1-5 likert scale for extraversion.
- A_Scale_score: the average response on a 1-5 likert scale for agreeableness.
- O_Scale_score: the average response on a 1-5 likert scale for openness.
- C_Scale_score: the average response on a 1-5 likert scale for conscientiousness.
- N_Scale_score: the average response on a 1-5 likert scale for neuroticism.

## Dataset

This provides you with information on which dataset the respondent ID was included in for the purposes of the competition 

- Train: participant ID and full data was given to each competition participant.
- Dev: participant ID and open-ended responses were given to each competition participant and they were asked to predict each trait score. This was used as the **Public Leaderboard**
- Test: participant ID and open-ended responses were given to each competition participant and they were asked predict each trait score. This was used as the **Private Leaderboard**
