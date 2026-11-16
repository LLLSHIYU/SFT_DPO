Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Precise as in it captures the most important points in the post. Concise as in it does not include unimportant or irrelevant details. It should be a good TL;DR of the post - so a good summary should be no more than 2-3 sentences.

Example:

Post: I [18M] know her [15F] now over a Year (since I know my friend [17M]). We haven't really talked to much I saw her from time to time when I was at his place over the weekend. About 2 months ago we came a little closer (I was at his place again and we played on his Xbox One when he went to bed early so we played Dance Central [Dancing Game] until 3am). Since then we've talked more often, she hang out with us and 2 Weeks ago we began texting. Now I would really like to ask her out on a date. But I still haven't talked to my friend about this... I mean she is still his sister, wouldn't it be wierd, plus she is a little young in my opinion(she's her age way ahead in mind)."

Summary A: I'd like to date the sister of my best friend. But I don't know if she's to young and don't how to start all of this.

Summary B: You've known a 15-year-old girl for over a year through your 17-year-old friend. Recently, you've been texting and getting closer. You're considering asking her out on a date, but you're unsure if it's appropriate since she's your friend's sister and she's younger than you expect.

In this example, summary A is better since it is more concise and captures the most important points in the post.

Post: {post}

Summary A: {{ 'model': 'model_1', 'summary': {output_1} }}

Summary B: {{ 'model': 'model_2', 'summary': {output_2} }}

Now please rank the models by the quality of their summaries, so that the model with rank 1 has the best summary. Then return a dictionary of the form

{{
"reasoning": short justification of ranking,
"model_1": rank_1,
"model_2": rank_2
}}

Your response must be a valid json dictionary and should contain nothing else because we will directly execute it in Python.
