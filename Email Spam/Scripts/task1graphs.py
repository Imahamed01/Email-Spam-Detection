import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

def clean_text(text):
    return ' '.join(re.findall(r'\b\w+\b', text.lower()))

def plot_label_distribution_pie_and_bar(labels, output_path_pie, output_path_bar):
    label_counts = Counter(labels)

    plt.figure(figsize=(6, 6))
    plt.pie([label_counts[0], label_counts[1]], labels=['Ham (Legitimate)', 'Spam'], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    plt.title('Label Distribution (Ham vs. Spam) - Pie Chart')
    plt.savefig(output_path_pie)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(['Ham (Legitimate)', 'Spam'], [label_counts[0], label_counts[1]], color=['skyblue', 'salmon'])
    plt.ylabel('Count')
    plt.title('Label Distribution (Ham vs. Spam) - Bar Graph')
    plt.savefig(output_path_bar)
    plt.close()

def plot_word_count_distribution(messages, labels, output_path):
    ham_word_counts = [len(message.split()) for message, label in zip(messages, labels) if label == 0]
    spam_word_counts = [len(message.split()) for message, label in zip(messages, labels) if label == 1]

    plt.figure(figsize=(10, 6))
    plt.hist([ham_word_counts, spam_word_counts], bins=40, color=['skyblue', 'salmon'], label=['Ham', 'Spam'], edgecolor='black')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution in Messages (Ham vs. Spam)')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_subject_length_distribution(subjects, output_path):
    subject_lengths = [len(subject) for subject in subjects]

    plt.figure(figsize=(10, 6))
    plt.hist(subject_lengths, bins=40, color='purple', edgecolor='black')
    plt.xlabel('Subject Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Subject Lengths')
    plt.savefig(output_path)
    plt.close()

df = pd.read_csv('cleaned_messages.csv')

df['cleaned_message'] = df['message'].apply(clean_text)
df['cleaned_subject'] = df['subject'].apply(clean_text)

plot_label_distribution_pie_and_bar(df['label'], 'label_distribution_pie.png', 'label_distribution_bar.png')
plot_word_count_distribution(df['cleaned_message'], df['label'], 'word_count_distribution.png')
plot_subject_length_distribution(df['subject'].dropna(), 'subject_length_distribution.png')

print("Graphs generated and saved as images.")
