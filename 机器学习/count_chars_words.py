#用python实现：分别统计文本文件中的字符个数、单词个数；要求统计结果输出到另外一个文件中，源文件和结果文件名均由命令行参数指定。
import sys

def count_chars_words(input_file,output_file):
    char_num = 0
    word_num = 0
    with open(input_file, encoding ='utf-8') as fr:   #读文件
        for line in fr:
            char_num = char_num+len(line)    #统计字符个数
            word_num = word_num+len(line.split())    #统计单词个数，默认以空格作为分隔符

    with open(output_file, 'w',encoding ='utf-8') as fw:
        fw.write(f"字符个数为：{char_num}，单词个数为：{word_num}")

    print("Finished!")

if __name__ == '__main__':
    if len(sys.argv) !=3:
        print("Usage:python script.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        count_chars_words(input_file,output_file)

