Search.setIndex({docnames:["EVALUATION","README","hashformers","hashformers.beamsearch","hashformers.ensemble","hashformers.evaluation","hashformers.experiments","hashformers.segmenter","index","modules","reference/index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["EVALUATION.md","README.md","hashformers.rst","hashformers.beamsearch.rst","hashformers.ensemble.rst","hashformers.evaluation.rst","hashformers.experiments.rst","hashformers.segmenter.rst","index.rst","modules.rst","reference/index.rst"],objects:{"":[[2,0,0,"-","hashformers"]],"hashformers.beamsearch":[[3,0,0,"-","algorithm"],[3,0,0,"-","bert_lm"],[3,0,0,"-","data_structures"],[3,0,0,"-","gpt2_lm"],[3,0,0,"-","model_lm"],[3,0,0,"-","reranker"]],"hashformers.beamsearch.algorithm":[[3,1,1,"","Beamsearch"]],"hashformers.beamsearch.algorithm.Beamsearch":[[3,2,1,"","flatten_list"],[3,2,1,"","next_step"],[3,2,1,"","reshape_tree"],[3,2,1,"","run"],[3,2,1,"","trim_tree"],[3,2,1,"","update_probabilities"]],"hashformers.beamsearch.bert_lm":[[3,1,1,"","BertLM"]],"hashformers.beamsearch.bert_lm.BertLM":[[3,2,1,"","get_probs"]],"hashformers.beamsearch.data_structures":[[3,1,1,"","Node"],[3,1,1,"","ProbabilityDictionary"],[3,4,1,"","enforce_prob_dict"]],"hashformers.beamsearch.data_structures.Node":[[3,3,1,"","characters"],[3,3,1,"","hypothesis"],[3,3,1,"","score"]],"hashformers.beamsearch.data_structures.ProbabilityDictionary":[[3,3,1,"","dictionary"],[3,2,1,"","get_segmentations"],[3,2,1,"","get_top_k"],[3,2,1,"","to_csv"],[3,2,1,"","to_dataframe"],[3,2,1,"","to_json"]],"hashformers.beamsearch.gpt2_lm":[[3,1,1,"","GPT2LM"],[3,1,1,"","PaddedGPT2LMScorer"]],"hashformers.beamsearch.gpt2_lm.GPT2LM":[[3,2,1,"","get_probs"]],"hashformers.beamsearch.model_lm":[[3,1,1,"","ModelLM"]],"hashformers.beamsearch.reranker":[[3,1,1,"","Reranker"]],"hashformers.beamsearch.reranker.Reranker":[[3,2,1,"","rerank"]],"hashformers.ensemble":[[4,0,0,"-","top2_fusion"]],"hashformers.ensemble.top2_fusion":[[4,1,1,"","Top2_Ensembler"],[4,4,1,"","run_ensemble"],[4,4,1,"","top2_ensemble"]],"hashformers.ensemble.top2_fusion.Top2_Ensembler":[[4,2,1,"","run"]],"hashformers.evaluation":[[5,0,0,"-","modeler"],[5,0,0,"-","utils"]],"hashformers.evaluation.modeler":[[5,1,1,"","Modeler"]],"hashformers.evaluation.modeler.Modeler":[[5,2,1,"","calculateAccuracy"],[5,2,1,"","calculateFScore"],[5,2,1,"","calculatePrecision"],[5,2,1,"","calculateRecall"],[5,2,1,"","calculateScore"],[5,2,1,"","countEntry"],[5,2,1,"","getRunCode"],[5,3,1,"","hashtagSegmentor"],[5,2,1,"","isFeatureOn"],[5,2,1,"","loadModelerParams"],[5,2,1,"","loadParameter"],[5,2,1,"","loadParameters"],[5,3,1,"","modelerParams"],[5,3,1,"","n"],[5,3,1,"","p"],[5,3,1,"","r"],[5,2,1,"","reset"],[5,2,1,"","segmentFile"],[5,2,1,"","segmentHashtag"],[5,3,1,"","t"],[5,2,1,"","test"],[5,3,1,"","totalh"],[5,3,1,"","totals"],[5,2,1,"","train"]],"hashformers.evaluation.utils":[[5,4,1,"","evaluate_dictionary"]],"hashformers.experiments":[[6,0,0,"-","evaluation"],[6,0,0,"-","utils"]],"hashformers.experiments.evaluation":[[6,4,1,"","evaluate_df"],[6,4,1,"","filter_top_k"],[6,4,1,"","read_experiment_dataset"]],"hashformers.experiments.utils":[[6,4,1,"","build_ensemble_df"],[6,4,1,"","calculate_diff_scores"],[6,4,1,"","filter_and_project_scores"],[6,4,1,"","project_scores"]],"hashformers.segmenter":[[7,0,0,"-","segmenter"]],"hashformers.segmenter.segmenter":[[7,1,1,"","BaseWordSegmenter"],[7,1,1,"","TweetSegmenter"],[7,1,1,"","TwitterTextMatcher"],[7,1,1,"","WordSegmenterCascade"]],"hashformers.segmenter.segmenter.BaseWordSegmenter":[[7,2,1,"","segment"]],"hashformers.segmenter.segmenter.TweetSegmenter":[[7,2,1,"","build_hashtag_container"],[7,2,1,"","compile_dict"],[7,2,1,"","extract_hashtags"],[7,2,1,"","replace_hashtags"],[7,2,1,"","segment"],[7,2,1,"","segmented_tweet_generator"]],"hashformers.segmenter.segmenter.WordSegmenterCascade":[[7,2,1,"","generate_pipeline"],[7,2,1,"","segment"]],hashformers:[[3,0,0,"-","beamsearch"],[4,0,0,"-","ensemble"],[5,0,0,"-","evaluation"],[6,0,0,"-","experiments"],[7,0,0,"-","segmenter"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0":[0,3,4,5,7],"00":0,"03213":1,"04":0,"06":0,"1":[0,3,4],"10":[1,5],"100":0,"1000":3,"11":0,"111":4,"12":0,"13":[0,3],"15":0,"15gb":0,"16gb":0,"2":[0,1,3,4],"20":[0,3],"2000":[],"2021":1,"2112":1,"222":4,"29":0,"3":[0,1],"32":0,"35":0,"36":0,"38":0,"41":0,"43":0,"44":0,"4405":0,"45":0,"47":0,"48":0,"50":0,"51":0,"55":0,"56":0,"58":0,"60":0,"63":0,"64":0,"65":0,"68":0,"7":1,"71":0,"72":0,"73":0,"74":0,"80":0,"81":0,"83":0,"86":0,"93":0,"case":3,"class":[3,4,5,7],"default":0,"do":1,"float":3,"import":1,"int":[],"return":[],"true":7,"try":0,"while":0,A:[0,1,7],As:0,For:1,If:1,In:0,It:1,On:1,The:0,_:0,a_diff:4,a_rank:4,about:0,accur:1,accuraci:8,acquila:1,across:0,ad:1,adjust:0,agnost:1,akira:1,alexandr:1,algorithm:[0,2,8,9,10],all:0,alpha:4,also:[0,1],altern:0,although:0,amount:[],analysi:1,ani:[1,7],anna:1,annot:0,anoth:1,api:[7,8],ar:[0,1],archiveprefix:1,arg:[3,5],argument:0,art:1,arxiv:1,astyp:3,author:1,automat:1,avail:0,averag:[0,1],b:6,b_diff:4,b_rank:4,base:[0,1,3,4,5,7],base_segment:7,basesegment:7,basewordsegment:7,basic:8,batch_siz:0,beamsearch:[0,2,8,9,10],below:[],bert:[0,1,3],bert_lm:[2,8,9,10],bertlm:3,best:[0,1],beta:4,better:0,between:[0,1],bool:7,both:0,bottom:0,boun:0,bring:0,build:[],build_ensemble_df:6,build_hashtag_contain:7,calculate_diff_scor:6,calculateaccuraci:5,calculatefscor:5,calculateprecis:5,calculaterecal:5,calculatescor:5,calixto:1,can:[0,1],candid:[],cascad:[],cascade_nod:7,cd:1,certain:[],charact:3,characters_field:[3,6],chave:1,citat:8,cl:1,clone:1,close:0,coerce_segmenter_object:[],colab:[0,1],cold:1,com:1,combin:0,compar:0,comparison:0,compat:1,compile_dict:7,confid:[],connect:0,consum:[],content:[9,10],contribut:8,control:[],core:[],countentri:5,cs:1,cu110:1,cuda:[1,3],current:1,danta:1,data:[3,5,6],data_structur:[2,8,9,10],datafram:[],dataset:[0,1,3,6],decreas:[],deep:0,deleteencodinglay:[],detail:[0,1],detect:[],dev:0,develop:1,devic:3,df:6,dict:[3,7],dict_1:4,dict_2:4,dictionari:3,differ:0,directli:1,distant:0,distilgpt2:0,doc:1,document:1,doesn:0,domain:0,don:0,e:1,either:0,ekphrasi:0,ekphrasiswordsegment:[],enforce_prob_dict:3,english:1,ensembl:[2,7,8,9,10],ensemble_rank:[],ensembler_kwarg:7,environ:1,eprint:1,estim:[],evalu:[2,8,9,10],evaluate_df:6,evaluate_dictionari:5,even:0,expect:0,experi:[0,2,8,9,10],extract_hashtag:7,face:1,fallback:[],fals:[3,6,7],faster:0,featur:5,featurefil:5,featurefilenam:5,few:0,figur:0,filenam:3,filepath:3,filetoseg:5,fill:[3,6],filter_and_project_scor:6,filter_top_k:6,find_seg:[],first:0,five:0,fix:0,flag:7,flatten_list:3,follow:1,fork:1,frame:[],framework:[0,1],from:[0,1],gener:7,generate_pipelin:7,get:0,get_prob:3,get_segment:3,get_top_k:3,getruncod:5,git:1,github:1,given:[],gold:[5,6],gold_arrai:3,gold_field:6,gome:1,googl:[0,1],gpt2:[0,1,3],gpt2_lm:[2,8,9,10],gpt2lm:3,gpt2lmscorer:3,gpt:1,gpu:0,gpu_batch_s:3,gpu_id:3,gram:0,had:0,hashform:[0,10],hashset:[0,1],hashtag:[0,1,5,6,7],hashtag_charact:7,hashtag_set:7,hashtag_token:7,hashtagcontain:[],hashtagmast:0,hashtagsegmentor:5,have:0,here:1,highlight:0,how:0,howev:[],http:1,hub:1,hug:1,hugo:1,hypothesi:3,iacer:1,ic:1,icecold:1,impact:0,index:8,inform:0,inner:1,input:[],input_df:6,instal:8,inuzuka:1,isfeatureon:5,juliana:1,just:[0,1],k:[3,6],know:0,known:0,kwarg:[3,7],languag:[0,1],larg:0,larger:[],layer:0,layer_list:[],learn:0,less:[],librari:[0,1],lie:[],list:7,list_:3,list_of_candid:3,literatur:0,lm_scorer:3,loadmodelerparam:5,loadparamet:5,log:[],lower:7,lowercas:0,magnitud:0,mai:0,mani:0,manual:0,marcelo:1,matcher:7,measur:3,memori:0,method:0,misc:1,model:[0,1,2,3,6,8,9,10],model_lm:[2,8,9,10],model_name_or_path:3,model_typ:3,modelerparam:5,modellm:3,modul:[8,9,10],more:[0,1],mpnr:0,multilingu:1,must:1,mxnet:1,n:[0,5],name:0,nascimento:1,nation:1,need:1,never:[],next_step:3,node:3,none:[1,3,5,7],nonetyp:[],number:[],object:[3,4,5,7],off:0,onli:0,option:7,order:0,our:1,output:[],p:5,packag:[1,8,9,10],paddedgpt2lmscor:3,page:8,panda:[],paper:[0,8],param:5,paramet:0,park:1,pass:0,per:0,perform:0,pip:1,portion:0,possibl:1,predict:[],preprocessing_kwarg:7,prev:[],primaryclass:1,print:1,prob_dict:3,probabilitydictionari:3,project_scor:6,provid:0,prune_segmenter_lay:[],pull:1,purpos:7,py:0,python:1,r:5,ram:0,rang:[],rank:[],reach:0,read:1,read_experiment_dataset:6,reason:[],refer:8,regex_flag:7,regex_pattern:7,regex_rul:[],regexwordsegment:[],relev:8,reli:0,remark:0,replac:1,replace_hashtag:7,replacement_dict:7,repositori:1,reproduc:0,request:1,rerank:[0,1,2,7,8,9,10],reranker_gpu_batch_s:[],reranker_kwarg:7,reranker_model_name_or_path:1,reranker_model_typ:[],reranker_rank:[],reranker_run:4,research:0,reset:5,reshape_tre:3,respland:1,result:0,return_datafram:3,return_rank:7,rocha:1,rodrigu:1,rodrigues2021zeroshot:1,row:0,ruan:1,ruanchav:1,rule:[],run:[3,4],run_ensembl:4,s:1,sampl:0,sant:1,santo:1,score:[3,6],score_field:[3,6],script:0,search:8,second:[0,1],segment:[0,1,2,3,5,6,8,9,10],segment_word:[],segmentation_field:[3,6],segmentation_gener:[],segmented_tweet_gener:7,segmenter_devic:[],segmenter_gpu_batch_s:[],segmenter_kwarg:7,segmenter_model_name_or_path:1,segmenter_model_typ:[],segmenter_rank:[],segmenter_run:[4,7],segmentfil:5,segmenthashtag:5,sentiment:1,separ:7,set:0,shot:1,sidenot:0,signific:0,so:0,sota:0,space:1,speed:8,stanford:0,state:1,statist:0,step:[0,1,3],str:[3,7],string:[],submodul:[2,8,9,10],subpackag:[8,9,10],t4:0,t:[0,5],tabl:0,take:0,task:1,telescop:[],tesla:0,test:5,testfil:5,text:[],than:[0,1],thei:0,thi:[0,1],threshold:[],time:[],titl:1,to_csv:3,to_datafram:3,to_json:3,togeth:0,top2_ensembl:4,top2_fus:[2,8,9,10],top:[],topk:[0,3],total:5,totalh:5,trade:0,train:5,transform:0,transformerwordsegment:1,tree:3,trim_tre:3,truesegment:5,tutori:1,tweet:7,tweetsegment:7,tweetsegmenteroutput:[],twittertextmatch:7,type:[],uncas:1,under:0,union:[],unstabl:0,up:0,update_prob:3,us:[0,1],usag:[0,8],use_ensembl:7,use_rerank:7,usual:0,util:[0,2,8,9,10],valu:0,version:[0,1],wa:0,want:1,we:[0,1],weight:[],welcom:1,weneedanationalpark:1,were:0,when:[],where:[],whether:[],which:[0,1],without:1,word:[0,1,7],word_list:7,word_segment:7,word_segmenter_output:[],wordsegment:[0,1],wordsegmentercascad:7,wordsegmenteroutput:[],work:1,world:0,ws:1,year:1,you:[0,1],your:1,zero:1},titles:["Evaluation","\u2702\ufe0f hashformers","hashformers package","hashformers.beamsearch package","hashformers.ensemble package","hashformers.evaluation package","hashformers.experiments package","hashformers.segmenter package","Welcome to hashformers\u2019s documentation!","hashformers","API Reference"],titleterms:{accuraci:0,algorithm:3,api:10,basic:1,beamsearch:3,bert_lm:3,citat:1,content:[2,3,4,5,6,7,8],contribut:1,data_structur:3,document:8,ensembl:4,evalu:[0,5,6],experi:6,gpt2_lm:3,hashform:[1,2,3,4,5,6,7,8,9],indic:8,instal:1,model:5,model_lm:3,modul:[2,3,4,5,6,7],packag:[2,3,4,5,6,7],paper:1,refer:10,relev:1,rerank:3,s:8,segment:7,speed:0,submodul:[3,4,5,6,7],subpackag:2,tabl:8,top2_fus:4,usag:1,util:[5,6],welcom:8}})