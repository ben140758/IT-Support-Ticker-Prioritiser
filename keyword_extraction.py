from nlp_functionality import *

from nltk import word_tokenize
import yake


# RAKE
#rake_nltk_var = Rake()


# YAKE
kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 2
deduplication_threshold = 0.3
numOfKeywords = 5

def myFunc(e):
    return e[1]

def extractKeywordsWithYAKE(text):
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    hi = kw_extractor.extract_keywords(text.lower())
    
    hi.sort(key=lambda a: a[1])
    tokenized = [item for sublist in [word_tokenize(y) for y in [x[0] for x in hi]] for item in sublist]
    tokenized2 = []
    for x in tokenized:
        if x not in tokenized2:
            tokenized2.append(x)
    ##print(tokenized2)
    #[print(x) for x in hi]
    return tokenized2
    
    #print("\n\n\n-------------------------        --------------------------\n\n\n")
    #ppd = preProcessText(text)
    #ppd2 = kw_extractor.extract_keywords(" ".join(ppd))
    #[print(x) for x in ppd2]
    
"""
def extractKeywordsWithRAKE(text):
    ppd = preProcessText(text)
    rake_nltk_var.extract_keywords_from_text(" ".join(ppd))
    ranked_keywords = rake_nltk_var.get_ranked_phrases()
    tokenized = [word_tokenize(y) for y in [x[0] for x in ranked_keywords]]
    [print(x[0]) for x in ranked_keywords]
    print(tokenized)
    print("\n\n+++++++++++++++++++++++++++++++\n\n")

def TF(text):
    new_dict = {}
    pp_text = preProcessText(text)
    for word in pp_text:
        try:
            new_dict[word] += 1
        except KeyError:
            new_dict [word] = 1
    new_dict1 = dict(sorted(new_dict.items(), key=lambda item: item[1], reverse=1))
    new_dict2 = {}
    tempwords = []
    tempnums = []
    for key in new_dict:
        if len(tempwords) != numOfKeywords:
            tempwords.append(key)
            tempnums.append(new_dict1[key])
        else:
            if new_dict[key] > min(tempnums):
                tempindex = tempnums.index(min(tempnums))
                del tempnums[tempindex]
                del tempwords[tempindex]
                tempwords.append(key)
                tempnums.append(new_dict1[key])

    #print(tempnums)
    for index, x in enumerate(tempwords):
        new_dict2[x] = tempnums[index]
    print(new_dict2)
    return new_dict2
    

def TF_YAKE(keywords, TF_dict):
    crossover = []
    

    for key in TF_dict:
        for word in keywords:
            if key in word[0]:
                crossover.append(word[0])
    print(crossover)
"""
if __name__ == "__main__":

    text = "To whom it may concern,                                            When going to log into my  university email today, it said that I needed to approve a request on my  authenticator app. So I downloaded the app and logged in but then it asked  for my phone number to send me a verification code so I put in my number  and waited. But I realised I had no service so no code was sent. So I  turned my phone on and off again to regain service and eventually the code  came through. Then when I went back onto the app there was no where to put  the code and i was on what I assume is the home page, Ive added  screenshots below. So I assumed I was somehow logged in. But then I went to  log into my uni email and it said it had sent a request, I got no request  in the app and it said on my laptop that my request was denied, even though  I saw no request. So I cant access my uni email. I then spoke to a member  of the it team at Microsoft and they basically said there was nothing they  could do and the university it team would be the best people to help. Ill  put the screen shots of what he said below. So now Im just wondering if  you could please help me cause I cannot access my uni emails now"
    #extractKeywordsWithRAKE(text)
    keywords = extractKeywordsWithYAKE(text)
    new_dict = TF(text)
    
    print("\n\n\n---------------------------------------------------\n\n\n")
    text = """Issues with the desktop computers - unable to download windows update - my 
 colleague and I have had messages pop up on our computers 
 to say the windows update wasn't downloaded and I've attached a screenshot 
 of the message we get when we select more info.
 
 Additionally, we have little storage on our devices that affect our daily 
 use of onedrive and teams.
 
 Thank you very much for all your help with this"""
    extractKeywordsWithYAKE(text)
    TF(text)
    print("\n\n\n---------------------------------------------------\n\n\n")
    text = """"I'm having trouble logging in to my E vision as it's saying my password is 
 incorrect so I'm therefore having trouble re-enrolling. 
 
 If there's anything you guys can do to help, I would greatly appreciate it. 
 
 
 All the best, """

    extractKeywordsWithYAKE(text)

    print("\n\n\n---------------------------------------------------\n\n\n")
    extractKeywordsWithYAKE("Got issues with microsoft office, cannot save word document, need this doing for tomorrow")
    TF(text)
    


                    
