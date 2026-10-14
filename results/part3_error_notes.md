# Part III error notes (assistant draft, written as an explicit exception to the project rule)

**Status:** drafted by the assistant on 2026-10-14 at the user's request. This is NOT your text: check it against `results/part3_error_sample_<variant>.csv`, rewrite it in your own words, and put what you keep in your own *My notes* block in `RESULTS.md` (that block was not touched). Reading is by one pass over the two 15-review samples, so every pattern below is a tendency in 30 reviews, not a measured rate. Counts of "n of 15" are my manual classification; a review can fit more than one pattern.

## What was read

- 15 random misclassified test reviews per variant (seed 0): `p3_feat_lstm_mean` (904 errors in total) and `p3_feat_transformer_mean` (1,128 errors in total). The reviews were displayed cut at 900 characters, so for truncated reviews I did not see the part after token 256 or the end of long ones.
- Counted properties for all errors vs all correct reviews are in `results/part3_error_stats.csv`.



## Patterns in the LSTM sample (15 errors)

1. **Positive review with complaints or negative-sounding words, overall verdict positive (about 8 of 15).** Examples: idx 23617 ("There is no gore and nudity... the script takes too much time... this is its major flaw... 7 out of 10"), 39703 ("bad sound/editing/lighting/acting... However... it is really good"), 30809 ("the acting is a bit chewy", "crappy spin off", "don't watch that movie", then "it would be a winner"), 33556 ("dated terribly... stilted... but it's still interesting"), 17772, 39476, 49626, 33599. All were predicted negative.
2. **Negative review that praises first or praises parts (about 4 of 15).** idx 37791 ("Unashamedly ambitious", "starts well enough", then "just not that good"), 43208 (several positive phrases: "treasures", "nice turns", "dynamic" next to "drably directed", "weak script"), 23894 (long plot summary, then criticism; truncated), 31882 (mostly factual description of a film, few opinion words). All were predicted positive.
3. **Anger or frustration aimed at something other than the film (2 of 15).** idx 43779 (annoyed that a network cancelled a show the writer loved; predicted negative) and idx 39973 (annoyed at people who rated the film low; predicted negative).
4. **Short review (1 of 15):** idx 9411, 74 tokens, a Columbo review with "you would have expected criminals to know...", "for some reason the murderer still doesn't pay enough attention", then "wonderful scenes". Predicted negative.



## Patterns in the Transformer sample (15 errors)

1. **Sarcasm, irony or deadpan mockery in negative reviews (about 4 of 15).** idx 10082 ("I recommend this film if you are suicidal"), 18606 ("Wonderful lines such as...", "the scene where the baby gets shot will make you cry (with laughter)"), 15044 ("This movie is very scary... We got this movie free with a pizza. You get what you pay for."; 72 tokens), 11018 (jokes about "TV hack-tors"). All predicted positive.
2. **Praise before or around a negative verdict (about 4 of 15).** idx 5307 (the story is "a masterwork", the film "a mediocrity", some scenes "beautiful"), 23294 ("fine cast", "beautifully composed" visuals, then "too contrived for my taste"), 48238 ("charming", "perfect movie ending" inside a complaint), 182 (elaborate, formal wording; truncated). Predicted positive.
3. **Lukewarm or hedged review (about 2 of 15).** idx 19575 ("more or less acceptable", "exquisite slaughtering", "very idiotic"; true negative) and idx 8794 ("it's ok", "not bad", "turn-off", then "a good movie", "7/10"; true positive).
4. **Positive review that spends words on faults or negative-sounding plot events (about 5 of 15).** idx 48419 ("acting was pretty bad", "not much more than an extended music video", then "very entertaining"), 10953 ("let down", "weak", "sadly", then "well worth a look"), 8586 (comedy about a house project: "blown to bits", "excessive", "somewhat thin", then "Hilarious!"), 10900, 40470 (argues against another reviewer: "Say, WHAT? No way. NOTHING like..."; truncated). Predicted negative.



## Your four checklist items

- **Negation.** Negation words are about as common in correct reviews as in errors (share of reviews with at least one: LSTM errors 87.7% vs correct 87.4%; Transformer errors 89.3% vs correct 87.0%), so their presence alone does not separate errors from correct reviews. In the samples negation appears inside both positive reviews ("There is no gore", "never an indication...") and negative ones ("just not that good", "I do not see what is the whole deal"), so I could not link the errors to negation in particular. A count of negation words cannot tell whether the negation changes the sentiment.
- **Length.** Errors are slightly longer than correct reviews (mean tokens kept: LSTM 203.8 vs 195.1; Transformer 202.0 vs 195.1), and more of them were truncated at 256 tokens (LSTM 45.0% vs 37.0%; Transformer 41.8% vs 37.5%). In the samples, 4 of 15 (LSTM) and 6 of 15 (Transformer) errors were truncated. Short reviews also appear (74 and 72 tokens), so errors are not only long reviews.
- **Unknown words.** The `<unk>` rate is slightly higher in errors (LSTM 2.34% vs 2.02%; Transformer 2.29% vs 2.02%). It varies from 0 to 10 `<unk>` tokens in the samples, including errors with 0. One Transformer error (idx 19575) has run-together words from missing spaces ("acceptableBloodline", "loteither", "ororiginal"), which cannot be in the vocabulary; I did not check which words became `<unk>` in the other reviews.
- **Sarcasm.** Present in about 4 of the 15 Transformer errors and not clearly in the LSTM sample (the "comic misadventures" and "heroes" quotes in idx 23894 are the closest).



## Things the samples do not show

- The two error sets overlap but are far from identical: of the full test errors, 563 reviews were missed by both classifiers, 341 only by the LSTM, 565 only by the Transformer.
- Across all errors the direction is roughly balanced (LSTM: 469 negative predicted positive, 435 positive predicted negative; Transformer: 536 and 592). The skew in the samples (LSTM sample 11 of 15 true positive; Transformer sample 9 of 15 true negative) is therefore sampling noise, not a finding.



## Perplexity ranking vs classification ranking (facts, from `results/part3_ranking.csv`)


| Model       | Part I test perplexity (lower is better) | Rank | Part III test accuracy / macro-F1 | Rank |
| ----------- | ---------------------------------------- | ---- | --------------------------------- | ---- |
| LSTM        | 80.45                                    | 1    | 0.8192 / 0.8192                   | 1    |
| Transformer | 156.94                                   | 2    | 0.7744 / 0.7744                   | 2    |


- The two rankings agree. Only two models were classified (the Part I RNN, perplexity 109.37, and the Part II frozen-embedding LSTMs were not), so agreement between two positions says little about whether better next-token prediction leads to better classification.
- The accuracy gap is 0.0448 on 5,000 test reviews (the binomial standard error of one accuracy near 0.8 is about 0.006). One run, seed 0, one pooling (mean), one classifier family.
- Other things differ between the two feature sets besides the language model: size (256 vs 100 columns) and the layer the vector comes from (last LSTM layer vs Transformer after its final LayerNorm), and the classifier chose different C (0.01 vs 0.1). These cannot be separated with the current runs.
- Models were trained on 64-token windows but features come from 256-token inputs (the Transformer's positions 64 to 255 were never updated in training, `DECISIONS.md` line 497). This applies to the Transformer features only and may matter when you compare the two.

