import os

def take_sen(f_name):
    f = open(f_name)
    sens = f.readlines()
    d = {}
    for el in sens:
        d[el[:8]] = el[11:]
    return d

def analyze_first_level(first_level_first, first_level_second, matched, tot):
    first_figs = first_level_first.split("and")
    for fig in first_figs:
        tot += 1
        # replace leading "a ...." or " a .... "
        if fig.startswith("a "):
            fig = fig[2:]
        if fig.startswith(" a "):
            fig = fig[3:]
        if fig.replace(" \n", "").replace("and", "") in first_level_second:
            if (fig.count("square") > 0 or fig.count("circle") > 0):
                matched += 1
    return matched, tot

def clean_string(fig,dict):
    for el in dict.keys():
        if el in fig:
            fig = fig.replace(el,"")
            break

    if fig.startswith("  a"):
            fig=fig[4:]
    if fig.startswith(" a"):
        fig=fig[3:]
    if fig.startswith("a"):
        fig=fig[2:]
    fig = fig.replace(" \n", "").replace("\n","").replace("and", "")
    return fig, (dict[el]+1)

def analyze_second_third_level(first, second, matched, tot,matched_third,tot_third):
    # dictionary of level
    d_level = {'the first one containing': 0,'the second one containing':1,
               'the third one containing' :2,'the fourth one containing':3,  'the <unk> one containing':4 }
    first_figs = first.split(";")

    for breadth in first_figs:
        # for each branch of the tree
        figs = breadth.split("and")
        # for each leaf
        if figs!=['\n']:
            for fig in figs:
                if fig.count("square")>0 or fig.count("circle")>0:
                    # if and only if is really the description of a figure
                    tot+=1
                    figAndthird_level= fig.split(" in ")
                    fig = figAndthird_level[0]
                    fig,position = clean_string(fig,d_level)

                    try:
                        if fig in second.split("the ")[position].replace(";","").replace("\n",""):
                            matched+=1
                    except IndexError:
                        a=2

                    try:
                        third_level = figAndthird_level[1]
                        tot_third+=1
                        if third_level in second.split("the ")[position].replace(";","").replace("\n",""):
                            matched_third+=1
                    except IndexError:
                        a = 2


    return matched, tot, matched_third,tot_third

def first_in_second(first_d,second_d):
    tot_first = 0
    matched_first = 0
    tot_second = 0
    matched_second = 0
    matched_third = 0
    tot_third = 0
    imgs = list( first_d.keys())
    for el in imgs:
        if type(first_d[el])==list:
            first_d[el] = first_d[el][0]
        first_level_first = first_d[el].split(" : ")[0]

        if type(second_d[el])==list:
            second_d = second_d[el][0]
        first_level_second = second_d[el].split(" : ")[0]

        matched_first, tot_first = analyze_first_level(first_level_first, first_level_second, matched_first, tot_first)
        if len (first_d[el].split(" : ") )>1:
            second_level_first = first_d[el].split(" : ")[1]
            try:
                second_level_second = second_d[el].split(" : ")[1]
            except IndexError:
                second_level_second =""
            matched_second,tot_second, matched_third, tot_third=analyze_second_third_level(second_level_first,
                    second_level_second, matched_second, tot_second,matched_third,tot_third)


    try:
        tot_second,matched_second
    except ZeroDivisionError:
        matched_second=0;tot_second=0
    try:
        tot_third,matched_third
    except ZeroDivisionError:
        tot_third=0;matched_third=0
    return {'tot_first':tot_first,'matched_first':matched_first,'tot_second': tot_second,
            'matched_second':matched_second,'tot_third':tot_third,'matched_third':matched_third}


#migliore  emb_dim_500_rnn_units_300_beta_0.0_hidden_coeff_4_lambd_8_drop_rate_0.2_it=60_beam=True.txt

#  emb_dim_500_rnn_units_200_beta_0.0_hidden_coeff_4_lambd_65_drop_rate_0.3_it=270_beam=False.txt
# 2500 2500
# first level  5662 	 5037 	 0.8896149770399152
# second level  3228 	 986 	 0.30545229244114
# third level  197 	 93 	 0.4720812182741117
# first level  6284 	 5045 	 0.8028325907065563
# second level  7991 	 1015 	 0.12701789513202352
# third level  1063 	 94 	 0.08842897460018814

#  emb_dim_500_rnn_units_300_beta_0.0_hidden_coeff_4_lambd_5_drop_rate_0.2_it=240_beam=True.txt
#  first level  5662 	 4682 	 0.8269162839985871
# second level  2960 	 1153 	 0.389527027027027
# third level  185 	 145 	 0.7837837837837838
# first level  6284 	 4714 	 0.7501591343093571
# second level  7991 	 1175 	 0.14704042047303217
# third level  1063 	 148 	 0.13922859830667922

#   new_emb_dim_800_rnn_units_560_beta_0.0_hidden_coeff_0.4_lambd_0.65_drop_rate_0.2_it=60_beam=True.txt
# 2500 2500
# first level  5689 	 4268 	 0.7502197222710494
# second level  2933 	 763 	 0.26014319809069214
# third level  174 	 120 	 0.6896551724137931
# first level  6284 	 4260 	 0.6779121578612349
# second level  7991 	 775 	 0.09698410712051057
# third level  1063 	 125 	 0.11759172154280338

#    new_emb_dim_600_rnn_units_480_beta_0.0_hidden_coeff_0.4_lambd_0.65_drop_rate_0.2_it=120_beam=True.txt
# 2500 2500
# first level  5688 	 3985 	 0.7005977496483825
# second level  2993 	 971 	 0.3244236551954561
# third level  188 	 143 	 0.7606382978723404
# first level  6284 	 3994 	 0.635582431572247
# second level  7991 	 986 	 0.12338881241396571
# third level  1063 	 146 	 0.13734713076199437

#    new_emb_dim_800_rnn_units_560_beta_0.0_hidden_coeff_0.4_lambd_0.65_drop_rate_0.2_it=90_beam=True.txt
# 2500 2500
# first level  5691 	 4345 	 0.7634862062906344
# second level  2936 	 1038 	 0.35354223433242504
# third level  183 	 143 	 0.7814207650273224
# first level  6284 	 4324 	 0.6880967536600892
# second level  7991 	 1039 	 0.1300212739331748
# third level  1063 	 149 	 0.14016933207902163

#   new_emb_dim_600_rnn_units_480_beta_0.0_hidden_coeff_0.4_lambd_0.65_drop_rate_0.2_it=120_beam=True.txt
# 2500 2500
# first level  5688 	 3985 	 0.7005977496483825
# second level  2993 	 971 	 0.3244236551954561
# third level  188 	 143 	 0.7606382978723404
# first level  6284 	 3994 	 0.635582431572247
# second level  7991 	 986 	 0.12338881241396571
# third level  1063 	 146 	 0.13734713076199437

#  emb_dim_500_rnn_units_200_beta_0.0_hidden_coeff_4_lambd_65_drop_rate_0.3_it=390_beam=True.txt
# 2500 2500
# first level  5644 	 5023 	 0.8899716513111269
# second level  2896 	 1144 	 0.39502762430939226
# third level  183 	 97 	 0.5300546448087432
# first level  6284 	 5041 	 0.8021960534691279
# second level  7991 	 1149 	 0.14378676010511826
# third level  1063 	 98 	 0.09219190968955786

#  emb_dim_500_rnn_units_200_beta_0.0_hidden_coeff_4_lambd_65_drop_rate_0.3_it=420_beam=True.txt
# 2500 2500
# first level  5671 	 5074 	 0.8947275612766707
# second level  3130 	 1249 	 0.3990415335463259
# third level  191 	 101 	 0.5287958115183246
# first level  6284 	 5070 	 0.8068109484404837
# second level  7991 	 1260 	 0.157677387060443
# third level  1063 	 103 	 0.09689557855126998

#    emb_dim_500_rnn_units_300_beta_0.0_hidden_coeff_4_lambd_65_drop_rate_0.3_it=390_beam=False.txt
# 2500 2500
# first level  5672 	 4709 	 0.8302186177715092
# second level  3183 	 1230 	 0.38642789820923656
# third level  190 	 128 	 0.6736842105263158
# first level  6284 	 4695 	 0.7471355824315723
# second level  7991 	 1235 	 0.1545488674759104
# third level  1063 	 132 	 0.12417685794920037
#  emb_dim_500_rnn_units_300_beta_0.0_hidden_coeff_4_lambd_65_drop_rate_0.3_it=390_beam=True.txt
# 2500 2500
# first level  5672 	 4711 	 0.8305712270803949
# second level  3183 	 1231 	 0.3867420672321709
# third level  190 	 126 	 0.6631578947368421
# first level  6284 	 4697 	 0.7474538510502864
# second level  7991 	 1237 	 0.154799149042673
# third level  1063 	 130 	 0.12229539040451552




"""
def first_in_second(d1,split_d1,d2,split_d2):
    tot = 0
    matched = 0
    for el in d1:
        d1_first_level = d1[el].split(split_d1)[0]
        d1_figs = d1_first_level.split(" a ")
        #d1_figs[0]=d1_figs[0].replace("a","")
        d2_first_level = (d2[el].split(split_d2)[0]).replace("\n","")
        for el in d1_figs:
            tot+=1
            if el.startswith("a "):
                el = el[2:]
            if el.replace("and","").replace("\n","") in d2_first_level:
                matched+=1
    print(tot,matched,matched/tot)



def second_level(d1,d2):
    matched=0
    tot=0
    for el in d1:
        if d1[el] =="":
            a = 2
        level2 = d1[el].split("the")[1:]
        for level in level2:
            for fig in level.split("and"):
                fig = fig.split(" containing ")
                tot+=1
                if fig[-1].replace("\n","").replace(" ;","") in d2[el]:
                    matched+=1
    print(tot,matched,matched/tot)

file_n="emb_dim_500_rnn_units_300_beta_0.0_hidden_coeff_4_lambd_8_drop_rate_0.2_it=30_beam=False.txt"
preds_d = take_sen("/home/davide/valentia_galli/"+file_n)
refs = take_sen("/home/davide/valentia_galli/my_dataset_sentences2.txt")
print(len(preds_d),len(refs))

refs_d = {}
for el in preds_d.keys():
    refs_d[el] = refs[el]

print(len(refs_d))
first_in_second(preds_d,":",refs_d,":")
first_in_second(refs_d,":",preds_d,":")

refs_d2 = {}
for el in refs_d:
    try:
        refs_d2[el] = refs_d[el].split(" : ")[1].replace("\n","")
    except IndexError:
        continue
preds_d2 = {}
for el in preds_d:
    try:
        preds_d2[el] = preds_d[el].split(" : ")[1]
        a = 2
    except IndexError:
        preds_d2[el] = ""

second_level(preds_d2,refs_d2)
second_level(refs_d2,preds_d2)



def second_level(d1,d2):
	matched=0
	tot=0
	for el in d1:
		level2 = d1[el].split("the")
		for ell in level2:
			figs = ell.split(" and ")
			for fig in figs:
				tot+=1
				if fig.replace(" ;","") in d2[el] and fig!="":
					matched+=1
					print(fig.replace(" ;",""))
	print(tot,matched)
"""