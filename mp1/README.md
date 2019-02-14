# Machine Problem 1

## 1. Vector Space Model

### 1.1 Understand Zipf's Law

#### 1.1 - 1
Normalization implementation

```java
public String Normalization(String token) {
  // remove all non-word characters
  // please change this to removing all English punctuation

  // after tokenization, the actual punctuation should only appear in the either ends
  token = token.replaceAll("(^\\p{Punct}+|\\p{Punct}+$)", "");

  // convert to lower case
  token = token.toLowerCase();

  // add a line to recognize integers and doubles via regular expression

  // take care of formatted number with comma: 3,000.01
  Pattern numberPtn = Pattern.compile("\\d+(,\\d{3})*(\\.\\d+)?");
  Boolean isNumber = numberPtn.matcher(token).matches();

  // and convert the recognized integers and doubles to a special symbol "NUM"
  if (isNumber) {
    token = "NUM";
  }

  return token;
}
```

#### 1.1 - 2

The log-log scale plot of TTF

<img width="575" alt="TTF-plot" src="./TTF.png">

For the linear interpolation, the coefficient is `-1.43903409` and the intercept is `6.90280094`, which stands for `10^6.90280094` on y-axis.

#### 1.1 - 3

The log-log scale plot of DF

<img width="575" alt="DF-plot" src="./DF.png">

For the linear interpolation, the coefficient is `-1.41265775` and the intercept is `6.77386445`, which stands for `10^6.77386445` on y-axis.

TTF fits Zipf's law better than DF. The high-frequency words tend to occure multiple times in a long document. Document frequency makes these head works have exact the same weight with the words only appear once in a document. While the head words are heavily pubished, the tail words which rarely appear more than once in a document are not impacted. The above plots can also approve this. While both the linear interpolation can closely fit the center and tail parts of the dots, the head of DF has a much larger difference than TTF.

### 1.2 Construct a Controlled Vocabulary

#### 1.2 - 1

Restaurant specific stopwords

```
n't
good
food
NUM
great
and_the
order
of_the
it_was
time
this_place
wait
servic
back
it_s
in_the
friend
love
fri
on_the
and_i
the_food
delici
ve
sauc
restaur
i_was
do_n't
dish
eat
for_a
```

### 1.2 - 2

The controlled vocabulary size is `3569`. The bigrams are removed if either of the tokens is a stopword.

### 1.2 - 3

The IDF is caculated with log of base `e` and the total training documents is `38688`.

The top 50 N-grams

```
word  DF    IDF
----------------------
tabl	6758	2.74480366
make	6529	2.779276879
menu	6097	2.84773387
chees	6066	2.852831311
tast	6042	2.85679564
chicago	5826	2.893200089
amaz	5798	2.898017721
thing	5769	2.903031999
nice	5642	2.925292157
peopl	5454	2.959181508
pretti	5447	2.960465795
seat	5396	2.969872861
night	5298	2.988201421
perfect	4969	3.052312242
recommend	4917	3.062832274
drink	4825	3.08172014
flavor	4811	3.084625914
price	4660	3.11651545
bar	4658	3.116944727
meal	4654	3.117803834
small	4517	3.147682862
dinner	4500	3.151453525
favorit	4461	3.160157971
bit	4449	3.162851577
worth	4424	3.168486669
made	4210	3.218068319
enjoy	4113	3.241378293
side	4101	3.244300138
experi	4092	3.246497137
chicken	4072	3.25139671
long	4013	3.265991907
pizza	3980	3.274249185
top	3942	3.283842802
fresh	3942	3.283842802
star	3851	3.307198172
line	3836	3.311100872
serv	3782	3.325278068
review	3775	3.327130657
lot	3697	3.348009374
day	3682	3.352074974
plate	3674	3.354250071
atmospher	3635	3.364921954
sweet	3537	3.392252173
ll	3532	3.393666802
minut	3477	3.409361242
tasti	3458	3.414840712
ca	3374	3.439432131
staff	3273	3.46982414
expect	3272	3.470129717
visit	3244	3.478724005
```

The bottom 50 N-grams

```
word     DF    IDF
----------------------
whitefish	51	7.631463581
tomorrow	51	7.631463581
well-season	51	7.631463581
sop	51	7.631463581
fluke	51	7.631463581
seed_bun	50	7.651266222
flawless	50	7.651266222
spici_king	50	7.651266222
side_salad	50	7.651266222
plethora	50	7.651266222
al_dent	50	7.651266222
pickl_onion	50	7.651266222
beard	50	7.651266222
teriyaki_mayo	50	7.651266222
fun_night	50	7.651266222
boast	50	7.651266222
hunt	50	7.651266222
skate	50	7.651266222
liter_melt	50	7.651266222
sloppi_goat	50	7.651266222
stringi	50	7.651266222
conveni_locat	50	7.651266222
final_seat	50	7.651266222
email	50	7.651266222
bean_burger	50	7.651266222
turnov	50	7.651266222
peev	50	7.651266222
januari	50	7.651266222
rpm_italian	50	7.651266222
univers	50	7.651266222
tang	50	7.651266222
condit	50	7.651266222
belong	50	7.651266222
pretti_bad	50	7.651266222
singl_thing	50	7.651266222
style_deep	50	7.651266222
declin	50	7.651266222
church	50	7.651266222
spicier	50	7.651266222
meat-eat	50	7.651266222
lunch_spot	50	7.651266222
pizza_hut	50	7.651266222
small_size	50	7.651266222
border	50	7.651266222
cocktail_list	50	7.651266222
frost	50	7.651266222
munchi	50	7.651266222
sushi/sashimi	50	7.651266222
tuna_roll	50	7.651266222
observ	50	7.651266222
```


