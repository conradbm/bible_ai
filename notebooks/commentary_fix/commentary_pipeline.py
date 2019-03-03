#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Commentary Pipeline


# In[12]:


import requests
import re
import string
from bs4 import BeautifulSoup
import bs4
import pickle
from collections import defaultdict
import datetime


# In[13]:


## Helper Functions
#1. `get_all_inner_content`: Intent is to get specified content between a stopping_class.


# In[14]:


def get_all_inner_content(nested,
                          stopping_class="versiontext",
                          flag_return_property='href',
                          debug=False,
                          verbose=False):

        if verbose:
            print(nested)

        # Return text
        content=""

        # Items between `stopping_class` and the next `stopping_class` Tag.\n",
        sib=nested.nextSibling

        # If specified, will append a specific tags text between\n",
        # `stopping_class` and next `stopping_class` Tag.\n",
        flag_return_property_list=[]

        # Keep going until the next `Tag` you find has `href` as an attribute\n",
        # From stopping_class to next stopping_class\n",
        while True:
            if verbose:
                print(sib)
            if isinstance(sib, bs4.element.Tag):

                # Collect your attribute text and object\n",
                try:
                    #print(sib['href'])\n",
                    #print(sib.text)
                    #input(\"!@!#!#!#!@##!@#!#!@!@############\")\n",
                    flag_return_property_list.append((sib.text, sib[flag_return_property]))
                    #input(\"____\")
                except:
                    pass

                try:
                    if sib["class"][0]==stopping_class:
                        break
                except:
                    pass

                # Check if we are at the stopping class
                try:
                    if sib["class"][0]==stopping_class:
                        break
                except:
                    pass

                # We are not at the stopping class, and are a Tag, so collect text field.\n",
                content+=sib.text
            else:
                # We are not a Tag, so just add the raw text from our object.
                content+=sib

            # Proceed to next item in the chain
            sib=sib.nextSibling

            # If no one else in the chain, exit loop.
            if sib is None:
                break

        return content,flag_return_property_list
            


# ## Function Objectives\n
# """
# "Provided a base_prefix, and a next suffix, we can crawl the entire biblehub\n",
#     "website to collect their bible versions, data, and cross references. The\n",
#     "objective of this function is to be called in a loop-like fashion, updating the\n",
#     "`next_suffix` part of the url. This should always start with a / because we simply\n",
#     "want to do string addition and have a live url at any point. Makes things easier.\n",
#     "Expected output should look like below:\n"
# """
# 
# `{"NIV\":{\"Genesis 1:1\":(\"in the beginning god created ...\", [\"John 1:!\", \"Revelation 4:14, ...\"]),\n",
#     "        \"Genesis 1:2\":(\"...\",[...])\n",
#     "        ...}\n",
#     " \"KJV\":{...},\n",
#     " ...\n",
#     " }`

# In[ ]:


## Populate data from link


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:



def populate_data_from_link(version_dict,
                            relevantBook, 
                            relevantChapter, 
                            base_prefix, 
                            next_suffix, 
                            debug=False,
                           verbose=False):

    if verbose:
        print("Scraping text.")
    if debug:
        print("Scraping text.")
        input("...")
    # Get content
    link=base_prefix+next_suffix

    if verbose:
        print("Link: ", link)

    if debug:
        print("Link: ", link)
        input("...")
    
    tries=10
    keepGoing=True
    while keepGoing:
        try:
            page = requests.get(link, timeout=100)
            soup = BeautifulSoup(page.text, 'html.parser')
            keepGoing=False
        except Exception as e:
            print(e)
            tries-=1
        if tries < 0:
            input("Tried 10 times with 100 second timeouts, still didnt work.")
            print(link)
            exit()

    # Get verse
    verse=soup.find(id="topheading").text
    verse_title=" ".join(verse.split(" ")[1:-1])
    book=" ".join(i for i in verse_title.split(" ")[:-1])
    chapter= verse_title.split(" ")[-1].split(":")[0]
    verse= verse_title.split(" ")[-1].split(":")[1]
    
    if debug:
        print("Collecting data on ", verse)
        print("Book: ", book)
        print("Chapter: ", chapter)
        print("Verse: ", verse)

    if not book == relevantBook or not chapter == relevantChapter:
        return True, version_dict, next_suffix

    # Populate cross references\n",
    crossrefs=[]
    for nested in soup.findAll(attrs={'crossverse'}):
        crossrefs.append(nested.text)

    for nested in soup.findAll(attrs={'versiontext'}):
        content,_=get_all_inner_content(nested, verbose=verbose, debug=debug)
        version_dict[nested.text][verse_title]=[content, crossrefs, next_suffix]
        if verbose:
            print("Content: ", content)
            print("References: " ,crossrefs)
            print("Link: ", next_suffix)

        if debug:
            print("Content: ", content)
            print("References: " ,crossrefs)
            print("Link: ", next_suffix)
            input("...")

    # Set up next page to crawl\n",
    nextLink=""
    for thing in soup.find(id={'topheading'}):
        if isinstance(thing, bs4.element.Tag):
            nextLink=thing.get_attribute_list('href')[0]

    if True:
        print("Next link ", str(nextLink))

    if debug:
        print("Next link ", str(nextLink))
        input("...")

    return False, version_dict, nextLink


# In[17]:


## Crawl link
# Provided a dictionary, a link, and a stopping point, we can crawl through the links on the page with the provided structure.


# In[18]:


def crawl_link(version_dict,
                   relevantBook,
                   relevantChapter,
                   base_prefix, 
                   next_suffix,
                   debug=False,
                   verbose=False):
    while True:
        done,version_dict,next_suffix=populate_data_from_link(version_dict,
                                                              relevantBook,
                                                              relevantChapter,
                                                              base_prefix,
                                                              next_suffix,
                                                              debug=debug,
                                                              verbose=verbose)

        if done:
            if verbose or debug:
                print("Done.", next_suffix)
            break
    return version_dict, next_suffix


# In[19]:


## Populate Commentary From Link
#Provided 2 dictionaries that need populated and the base and suffix links, we can crawl through each to populate data from the site for commentary <em>and</em> simple bible data with cross references


# In[ ]:


def populate_commentary_from_link(version_dict, 
                                      commentary_dict,
                                      base_prefix, 
                                      next_suffix,
                                      comment_base,
                                      comment_suffix,
                                      terminating=("Revelation", "22"),
                                      debug=False,
                                      verbose=False):
    if verbose:
        print("Scraping text.")
       
    if debug:
        print("Scraping text.")
        input("...")
    # Get content\n",
    link=comment_base+comment_suffix
    
    tries=10
    keepGoing=True
    while keepGoing:
        try:
            page = requests.get(link, timeout=100)
            soup = BeautifulSoup(page.text, 'html.parser')
            keepGoing=False
        except Exception as e:
            print(e)
            tries-=1
        if tries < 0:
            input("Tried 10 times with 100 second timeouts, still didnt work.")
            print(link)
            exit()

    # Get verse\n"
    verse=soup.find(id="topheading").text
    verse=" ".join(i for i in verse.split(" ")[1:-1])
    book=" ".join(i for i in verse.split(" ")[:-1])
    chapter=verse.split(" ")[-1]

    if True:
        print("Collecting data on ", verse)
        print("Book: ", book)
        print("Chapter: ", chapter)
        print("comment_base:", comment_base)
        print("comment_suffix:",comment_suffix)

    if debug:
        print("Collecting data on ", verse)
        print("Book: ", book)
        print("Chapter: ", chapter)
        print("comment_base:", comment_base)
        print("comment_suffix:",comment_suffix)
        input("...")
    
    if True:
        print("Crawling every verse within ", book ," ",chapter)
    version_dict,next_suffix=crawl_link(version_dict, book, chapter, base_prefix, next_suffix, verbose=verbose, debug=debug)
    
    # Get Scholarly Opinion And Attach to each verse here \n",

    if True:
        print("Gathering commentary for every verse within ", book, " ", chapter)
    for verse in soup.findAll(attrs={'versenum'}): 
        content,hrefs=get_all_inner_content(verse, stopping_class="versenum", verbose=verbose, debug=debug)
        commentary_dict[verse.text]=[content,hrefs]

        if verbose:
            print(commentary_dict[verse.text])
            print(hrefs)
            
        if debug:
            print(commentary_dict[verse.text])
            print(hrefs)
            input("..")
            
    # Get Next Link\n",
    nextLink=""
    for thing in soup.find(id={'topheading'}):
        if isinstance(thing, bs4.element.Tag):
            nextLink=thing.get_attribute_list('href')[0]
    comment_suffix=nextLink
    
    if True:
        print("Moving on to the next chapter.")
        #print("Version dict: ")
        #print(list(version_dict.items())[-1])
        #print("Commentary dict: ")
        #print(list(commentary_dict.items())[-1])
    
    if verbose:
        print("comment_suffix ", str(comment_suffix))
    
    if debug:
        print("comment_suffix ", str(comment_suffix))
        input("...")
    
    if book == terminating[0] and chapter == terminating[1]:
        return True, version_dict, commentary_dict, base_prefix, next_suffix, comment_base, comment_suffix
    return False, version_dict, commentary_dict, base_prefix, next_suffix, comment_base, comment_suffix


# In[21]:


## Crawl Commentary
# Objective is to extract content from the commentary page, which is structured by chapter. Hence we need to look in this first, then look into each individual link to get the content there, then populate it with the commentary for each.


# In[22]:


from collections import defaultdict
def crawl_commentary(version_dict=defaultdict(dict),
                         commentary_dict=dict(),
                         base_prefix="https://biblehub.com",
                         next_suffix="/genesis/1-1.htm",
                         comment_base="https://biblehub.com/commentaries/cambridge",
                         comment_suffix="../genesis/1.htm",
                        debug=False,
                        verbose=False):
        
    while True:
        comment_suffix=comment_suffix[2:]
        done,version_dict,commentary_dict,base_prefix,next_suffix,comment_base,comment_suffix=populate_commentary_from_link(version_dict,
                                                                                                                             commentary_dict,
                                                                                                                              base_prefix, 
                                                                                                                              next_suffix, 
                                                                                                                              comment_base,
                                                                                                                              comment_suffix,
                                                                                                                               debug=debug,
                                                                                                                               verbose=verbose)
        if done:
            print("Finished data collection")
            custom_out_name="bible_data_with_commentary_"+str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_")+".pkl"
            with open(custom_out_name, "wb") as handle:
                pickle.dump((version_dict, commentary_dict), handle)
            break
    return version_dict, commentary_dict


# In[ ]:


version_dict, comment_dict=crawl_commentary()


# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




