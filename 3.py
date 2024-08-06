import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    #Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    #Calculate word frequencies
    freq_dist = FreqDist(words)
    
    #Score each sentence based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        word_list = word_tokenize(sentence.lower())
        sentence_scores[sentence] = sum(freq_dist[word] for word in word_list if word in freq_dist)
    
    #Get the top N sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]
    
    #Combine the summary sentences into a single string
    summary = ' '.join(summary_sentences)
    
    return summary

def main():
    text = """
    The Impact of Technology on Modern Education
In the last few decades, technology has dramatically transformed the field of education. The introduction of digital tools has revolutionized the way students learn and teachers instruct. From interactive whiteboards to online courses, technology has brought significant changes to educational practices.
One of the major advancements in educational technology is the use of e-learning platforms. These platforms provide students with access to a vast array of resources and learning materials from anywhere in the world. Online courses and virtual classrooms have made education more accessible, allowing students to learn at their own pace and on their own schedule. This flexibility is particularly beneficial for those who cannot attend traditional in-person classes due to geographical or personal constraints.
Moreover, technology has facilitated personalized learning experiences. With the help of adaptive learning software, educators can tailor their teaching methods to meet the individual needs of each student. These tools analyze students' performance and provide customized feedback and resources, helping to address learning gaps and improve overall academic outcomes.
Another significant impact of technology on education is the enhancement of collaboration and communication. Digital tools such as video conferencing, collaborative documents, and educational apps have made it easier for students and teachers to interact and work together. Group projects and discussions can now be conducted online, breaking down barriers of time and space and fostering a more interactive and engaging learning environment.
However, the integration of technology in education also presents challenges. Issues such as digital divide, data privacy, and the need for teacher training are critical considerations. Not all students have equal access to technology, which can exacerbate educational inequalities. Additionally, the protection of students' personal information and the appropriate use of educational technology require careful attention.
In conclusion, while technology has undeniably brought numerous benefits to modern education, it is essential to address the accompanying challenges to ensure that all students can fully benefit from these advancements. As technology continues to evolve, its role in education will likely become even more prominent, shaping the future of learning in profound ways.
"""

    summary = summarize_text(text)

    print("Original Text:")
    print(text)
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()
