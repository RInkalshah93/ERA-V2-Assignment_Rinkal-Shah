# Assigment 18

## Obejective
1. Pick the "en-it" dataset from opus_book
2. Train transformer and get loss under 1.8 within 18 Epochs

## Introduction
The goal of the assigment to speed up the tranfomers training.

## Training Log

    Max length of source sentence: 309
    Max length of target sentence: 274
    Processing Epoch 00: 100%|██████████| 910/910 [03:12<00:00,  4.72it/s, loss=6.161]
    --------------------------------------------------------------------------------
    SOURCE: Oblonsky was wearing raw hide shoes, bands of linen wound round his feet instead of socks, a pair of tattered trousers and a short coat.
    TARGET: Stepan Arkad’ic aveva calzature di cuoio d’un sol pezzo, fasce, pantaloni sdruciti e cappotto corto.
    PREDICTED: — Ma , e il suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo .
    --------------------------------------------------------------------------------
    SOURCE: To find this out it was necessary to go further away in the direction of the wind.
    TARGET: Per trovare la direzione bisognava andare più lontano, sottovento.
    PREDICTED: Il suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo suo .
    --------------------------------------------------------------------------------
    Processing Epoch 01: 100%|██████████| 910/910 [03:11<00:00,  4.75it/s, loss=5.212]
    --------------------------------------------------------------------------------
    SOURCE: He stroked her, and whistled as a sign that she might now set off.
    TARGET: Levin l’accarezzò e fischiò in segno che si poteva cominciare.
    PREDICTED: Egli si fermò e si fermò , e si era un momento di più .
    --------------------------------------------------------------------------------
    SOURCE: 'Yet all over Europe education is now compulsory.'
    TARGET: — Ma in tutta Europa attualmente le scuole sono obbligatorie.
    PREDICTED: — Ma , è un ’ altra cosa è un ’ altra .
    --------------------------------------------------------------------------------
    Processing Epoch 02: 100%|██████████| 910/910 [03:11<00:00,  4.76it/s, loss=5.306]
    --------------------------------------------------------------------------------
    SOURCE: 'Love!' she thought, 'as if he can love!
    TARGET: Pensò: «Ama? Può forse amare lui?
    PREDICTED: — Come è così , come se è così , come se non lo può , come se non lo può , come se lo può , come se lo può , come se lo può , se lo può essere così !
    --------------------------------------------------------------------------------
    SOURCE: Then George tried taking it out with him late at night, and practising round the square.
    TARGET: Poi Giorgio tentò di portar fuori il banjo di notte e di esercitarsi nella piazzetta.
    PREDICTED: Dopo colazione , , e si a casa , e si .
    --------------------------------------------------------------------------------
    Processing Epoch 03: 100%|██████████| 910/910 [03:09<00:00,  4.80it/s, loss=5.401]
    --------------------------------------------------------------------------------
    SOURCE: "What beautiful ladies!" cried she in English. "Oh, I wish I might go to them!
    TARGET: — Oh che belle signore! — esclamò. — Come sarei contenta di andare con loro!
    PREDICTED: — Che cosa ! — esclamò , — mi disse , — che mi ho detto .
    --------------------------------------------------------------------------------
    SOURCE: Such stand simply elevated upon the goodwill and the fortune of him who has elevated them—two most inconstant and unstable things. Neither have they the knowledge requisite for the position; because, unless they are men of great worth and ability, it is not reasonable to expect that they should know how to command, having always lived in a private condition; besides, they cannot hold it because they have not forces which they can keep friendly and faithful.
    TARGET: Questi stanno semplicemente in sulla voluntà e fortuna di chi lo ha concesso loro, che sono dua cose volubilissime et instabili; e non sanno e non possano tenere quel grado: non sanno, perché, se non è uomo di grande ingegno e virtù, non è ragionevole che, sendo sempre vissuto in privata fortuna, sappi comandare; non possano, perché non hanno forze che li possino essere amiche e fedeli.
    PREDICTED: le cose che si e ' di quelle che si , e ' quali si , e ' loro , se non si , e ' tempi , perché non si , e ' loro tempi , e ' , e ' , e ' , e ' principi , non si a ' tempi , e ' populi , non si a ' principi , e ' principi , e ' principi , non si a ' principi , e ' , e ' principi , e ' principi , non si a ' principi , e ' principi , non si .
    --------------------------------------------------------------------------------
    Processing Epoch 04: 100%|██████████| 910/910 [03:11<00:00,  4.76it/s, loss=5.184]
    --------------------------------------------------------------------------------
    SOURCE: We did not want to overdo the thing and get a surfeit of it.
    TARGET: Non volevamo esagerare il divertimento e averne più del necessario.
    PREDICTED: Non ci .
    --------------------------------------------------------------------------------
    SOURCE: He had not meant to leave next day, but now decided to go away early in the morning.
    TARGET: Egli non aveva intenzione di partire il giorno dopo, ma ora decise di partire la mattina presto per tornare a casa.
    PREDICTED: Non aveva pensato a leggere , ma ora , ora , ora , in mattina .
    --------------------------------------------------------------------------------
    Processing Epoch 05: 100%|██████████| 910/910 [03:10<00:00,  4.78it/s, loss=4.212]
    --------------------------------------------------------------------------------
    SOURCE: 'But Society noticed, and that disturbs him! You are not well, Alexis Alexandrovich!' she added, rose and was about to pass out of the room, but he moved forward as if wishing to stop her.
    TARGET: «A lui personalmente non importa alcun che, ma la compagnia lo ha notato, e lui se ne inquieta». — Tu stai poco bene, Aleksej Aleksandrovic — aggiunse, alzandosi per uscire dalla porta; ma egli le si parò innanzi, quasi a fermarla.
    PREDICTED: — Ma la società , e quella che non si , non si , disse , Aleksej Aleksandrovic si alzò e si alzò , ma si alzò , come se volesse , si avvicinava a sé .
    --------------------------------------------------------------------------------
    SOURCE: The poor little thing was snorting like a steam-engine when she caught it, and kept doubling itself up and straightening itself out again, so that altogether, for the first minute or two, it was as much as she could do to hold it.
    TARGET: Il poverino quando Alice lo prese, ronfava come una macchina a vapore e continuava a contorcersi e a divincolarsi così che, per qualche istante, ella dubitò di non poterlo neanche reggere.
    PREDICTED: Il povero povero era così come un pasticcio di patate , quando si fu , e si mise a , e si , per due o due o due o tre o tre volte .
    --------------------------------------------------------------------------------
    Processing Epoch 06: 100%|██████████| 910/910 [03:10<00:00,  4.77it/s, loss=3.814]
    --------------------------------------------------------------------------------
    SOURCE: The Italian nurse, having smartened up the baby girl, came in and held her out to Anna.
    TARGET: La nutrice italiana, vestita la bambina, entrò con lei e la porse ad Anna.
    PREDICTED: La bambina , , si avvicinò alla bambina , si avvicinò a Anna e si accostò a Anna .
    --------------------------------------------------------------------------------
    SOURCE: He felt that instead of doing what he had meant to do and –warning his wife that she was making a mistake in the eyes of the world, he was involuntarily getting excited about a matter which concerned her conscience, and was struggling against some barrier of his imagination.
    TARGET: Si accorgeva che invece di quello che voleva fare, mettere cioè in guardia la moglie da quello che poteva apparire un errore agli occhi del mondo, si agitava involontariamente per quello che riguardava la coscienza di lei, e lottava contro un muro creato dalla sua stessa immaginazione.
    PREDICTED: Sentiva che , invece , si sforzava di ricordare quello che aveva fatto e che la moglie era fatto , che era in lui , si era di nuovo con gli occhi di lei , e si sforzava di calmarsi , di non essere di trovare in una luce nuova , che si era impossessata di sé e di sé .
    --------------------------------------------------------------------------------
    Processing Epoch 07: 100%|██████████| 910/910 [03:10<00:00,  4.78it/s, loss=3.139]
    --------------------------------------------------------------------------------
    SOURCE: Not a tie links me to any living thing: not a claim do I possess to admittance under any roof in England."
    TARGET: Nessun legame mi unisce a un essere vivente. Non ho il diritto di chiedere asilo sotto nessun tetto d'Inghilterra.
    PREDICTED: Non ho mai pensato a nulla ; non voglio in quei vantaggi che vi nell ' Inghilterra .
    --------------------------------------------------------------------------------
    SOURCE: I know you to be a really high-minded man,' said Betsy, stopping short in the little sitting-room and once again pressing his hand with peculiar warmth. 'I am only an outsider, but am so fond of her and respect you so much that I will take the liberty of advising you.
    TARGET: Io vi conosco per un uomo veramente generoso — disse Betsy fermandosi nel piccolo salotto e stringendogli in modo particolarmente forte ancora una volta la mano. — Io sono una persona estranea, ma voglio tanto bene a lei e stimo tanto voi che mi permetto darvi un consiglio.
    PREDICTED: So che tu sia un uomo alto — disse Betsy , fermandosi sulla stanza e , fermandosi sulla mano , con la mano , con la mano che gli . — Sono molto impacciata , ma sono così poco , e voi , credimi , che tu la libertà di voi .
    --------------------------------------------------------------------------------
    Processing Epoch 08: 100%|██████████| 910/910 [03:10<00:00,  4.79it/s, loss=3.506]
    --------------------------------------------------------------------------------
    SOURCE: It would take up a larger volume than this whole work is intended to be to set down all the contrivances I hatched, or rather brooded upon, in my thoughts, for the destroying these creatures, or at least frightening them so as to prevent their coming hither any more: but all this was abortive; nothing could be possible to take effect, unless I was to be there to do it myself: and what could one man do among them, when perhaps there might be twenty or thirty of them together with their darts, or their bows and arrows, with which they could shoot as true to a mark as I could with my gun?
    TARGET: Ma tutti questi erano aborti; niuno di tali disegni poteva avverarsi, finchè fossi stato qui io solo per mandarlo ad effetto. Che cosa un uomo poteva fare contro essi, che sarebbero forse stati in venti o trenta uniti insieme, che co’ loro dardi o con le loro frecce miravano giusto al segno, come avrei potuto far io col mio moschetto.
    PREDICTED: Sarebbe stato un ’ idea di tutto il lavoro più di tutto , che io tutto , o di tutto , io pensai a ogni possibile , o piuttosto a le cose che in appresso , se bene le ondate , come pure avessero costretti i loro modi : tal fine che avessero potuto , io non poteva permettere a chi fare altrimenti ; ma , se non mi fosse possibile , io , come un uomo che , se non mi , e che , se mi con un , o per , non poteva , e che , come fare uno in appresso , come in appresso , come in appresso , non aver avuto un ’ orma di un ’ orma di questi luoghi per me , non mi , e che , e che , come un ’ orma sul mio moschetto , non mi con un ’ orma sul mio moschetto per me , come un ’ orma sul mio moschetto , non mi , come un ’ orma di questi luoghi , come un ’ orma di questi nemici non fossero , e che fare altrimenti che fare altrimenti che , non mi , non mi , non mi , come io potessi , e che , come ho potuto , e che fare altrimenti che fare altrimenti che , non mi con un uomo , non mi con un ’ orma di questi luoghi ove con un ’ orma di tali modi di tali modi che fare altrimenti che , non mi , non mi , non mi , come ho potuto , come ho potuto , non mi , e che fare altrimenti che fare altrimenti che fare altrimenti che , come ho potuto fare altrimenti che , come ho potuto , come ho potuto , non mi , non mi , come fare altrimenti che fare altrimenti che , come un solo , come fare
    --------------------------------------------------------------------------------
    SOURCE: 'I can't stand it!' cried one of the red-faced landowners.
    TARGET: — Non ne posso più — diceva uno dei proprietari tutto rosso in viso.
    PREDICTED: — Io non posso entrare ! — gridò uno dei signori rossi del viso di palude .
    --------------------------------------------------------------------------------
    Processing Epoch 09: 100%|██████████| 910/910 [03:10<00:00,  4.79it/s, loss=3.054]
    --------------------------------------------------------------------------------
    SOURCE: But it was not her business to think for me, or to seek a place for me: besides, in her eyes, how doubtful must have appeared my character, position, tale.
    TARGET: Ma non era affar suo di pensare a me, o di trovarmi un posto; inoltre come dovevano esserle parsi equivoci il mio carattere, la mia posizione.
    PREDICTED: Ma non mi fu necessario pensare a me , o per cercare un luogo per me , per quanto per me gli occhi mi parvero e .
    --------------------------------------------------------------------------------
    SOURCE: Thou art not worth to me—no, not the taking off the ground; one of those knives is worth all this heap; I have no manner of use for thee—e’en remain where thou art, and go to the bottom as a creature whose life is not worth saying.”
    TARGET: Tu non mi giovi a nulla, affatto nulla! non compensi l’incomodo di levarti da terra; un di questi coltelli val più di tutto questo tuo mucchio. Di te non saprei in che modo servirmi; dunque resti dove sei, ed affondati col resto del vascello; tu non sei tal creatura, che meriti le sia salvata la vita.»
    PREDICTED: Non a me , non , non si ; uno dei due , le di queste : non ho il diritto di fare di fare alcuna importanza per me ; non si può , e alla vita che non si può dire come dire .
    --------------------------------------------------------------------------------
    Processing Epoch 10: 100%|██████████| 910/910 [03:10<00:00,  4.77it/s, loss=2.848]
    --------------------------------------------------------------------------------
    SOURCE: A dry sail clinging to your legs and wrapping itself round your head is not pleasant, but, when the sail is sopping wet, it becomes quite vexing.
    TARGET: Una vela asciutta, che vi s’aggrappa alle gambe e vi s’avvolge alla testa, non è piacevole; ma quando la vela s’è impregnata d’acqua diventa irritante.
    PREDICTED: Il pianista nervoso comincia da capo ad angolo e gli si voltano le mani ; ma quando la barca è piacevole , la vela , la .
    --------------------------------------------------------------------------------
    SOURCE: Mrs. Dent here bent over to the pious lady and whispered something in her ear; I suppose, from the answer elicited, it was a reminder that one of the anathematised race was present.
    TARGET: La signora Dent si curvò allora verso lady Ingram e le disse qualcosa a voce bassa. Suppongo che le facesse notare la presenza di una persona appartenente a quella razza, sulla quale aveva lanciato il suo anatema.
    PREDICTED: La signora Dent , dritto su la signora Fairfax , mi fissò per . — Suppongo che la risposta fosse , che la corsa fosse la facciata di casa .
    --------------------------------------------------------------------------------
    Processing Epoch 11: 100%|██████████| 910/910 [03:11<00:00,  4.76it/s, loss=2.405]
    --------------------------------------------------------------------------------
    SOURCE: Vronsky had come to the elections because he felt dull in the country, in order to proclaim to Anna his right to freedom, to repay Sviyazhsky by supporting him at these elections for all the trouble he had taken for Vronsky at the Zemstvo elections, and most of all to perform strictly all the duties of the position he had taken up as nobleman and landowner.
    TARGET: Vronskij era venuto alle elezioni perché si annoiava in campagna e perché aveva bisogno di affermare i propri diritti di libertà di fronte ad Anna, e per disobbligarsi con Svijazskij, appoggiandolo nelle elezioni, di tutte le beghe che s’era prese per Vronskij alle elezioni del consiglio; ma più di tutto per adempiere rigorosamente tutti gli obblighi di quella posizione di nobile e di proprietario di terre che s’era scelta.
    PREDICTED: Vronskij era venuto a considerare le elezioni , perché in campagna aveva idee tali per Anna la facoltà di far conoscere i mezzi per cultura , di cultura per quelle ragioni che aveva preso dalle api nelle elezioni precedenti di cultura , di Aleksej Vronskij e per tutti i più ricchi e per il più interessanti .
    --------------------------------------------------------------------------------
    SOURCE: His coarseness irritated her and gave her boldness. 'Is it possible you do not feel how easy it is for you to insult me?' she said.
    TARGET: La sua villania l’aveva irritata e le aveva dato coraggio. — Possibile che non sentiate come vi sia facile offendermi? — ella disse.
    PREDICTED: La sua collera e l ’ irritò , e come non vi neppure .
    --------------------------------------------------------------------------------
    Processing Epoch 12: 100%|██████████| 910/910 [03:09<00:00,  4.79it/s, loss=2.581]
    --------------------------------------------------------------------------------
    SOURCE: Ere long, I became aware that some one was handling me; lifting me up and supporting me in a sitting posture, and that more tenderly than I had ever been raised or upheld before.
    TARGET: Dopo poco mi accorsi che qualcuno si avvicinava a me, mi sollevava collocandomi in una posizione più comoda; nessuno mi aveva mai trattato fino a quel momento con tanta sollecitudine affettuosa.
    PREDICTED: Poco dopo mi sentii un ' altra cosa : mi si avvicinò a me e , più sulla roccia , più che mai mi fossi trovato in me , o più di prima , ero sbalordita .
    --------------------------------------------------------------------------------
    SOURCE: Suddenly a gentleman with sausage-shaped whiskers, and as red as a lobster, appears, announces that no one but his wife lives in that flat and turns them both out...'
    TARGET: Ed ecco, a un tratto compare il padrone di casa con le fedine a salsicciotto, rosso come un gambero, il quale spiega che in casa non c’è nessuno all’infuori di sua moglie, e li caccia via tutti e due.
    PREDICTED: Un entusiastico signore con le fedine , con le fedine più di una donna , come non sia , ma la moglie non ha in sé che tutti e due la .
    --------------------------------------------------------------------------------
    Processing Epoch 13: 100%|██████████| 910/910 [03:09<00:00,  4.80it/s, loss=2.306]
    --------------------------------------------------------------------------------
    SOURCE: Now when the thing is done... just consider, should one forsake her? Granted that one ought to part with her so as not to destroy one's family life, but oughtn't one to pity her and provide for her and make things easier?'
    TARGET: Ammettiamo pure: troncare per non distruggere la propria vita familiare; ma non si può forse avere pena di lei, provvedere, mitigare?
    PREDICTED: Adesso , quando è possibile la cosa ... la pensi , bisogna pensare ch ’ ella avrebbe dovuto con lei in modo tale da non mostrare la vita di una vita così grande , ma senza volere si per quello che le condizioni le chiedeva .
    --------------------------------------------------------------------------------
    SOURCE: However, at last, as I said, I made one to answer, and covered it with skins, the hair upwards, so that it cast off the rain like a pent-house, and kept off the sun so effectually, that I could walk out in the hottest of the weather with greater advantage than I could before in the coolest, and when I had no need of it could close it, and carry it under my arm.
    TARGET: In somma giunto, come ho detto, a farmene uno sufficientemente buono al mio scopo, lo copersi di pelle col pelo all’infuori; per tal modo parava da me l’acqua a guisa d’una grondaia, e mi difendeva sì efficacemente dal sole che poteva nella stagione più calda camminare attorno con maggior conforto di quanto avessi fatto dianzi nella più fresca. Se non aveva bisogno del mio arnese, me lo poneva, chiudendolo, sotto il braccio.
    PREDICTED: Pure , siccome ho già detto , lo poche parole di me ne scrisse una risposta , e quale sarei cioè costretto ad la pioggia come la pioggia . questa la pioggia , il sole era sì dura , e più difficile di quanti sforzi poteva cavare , al tempo di quella ad un tempo , che non era affatto al di fuori , se non avessi avuto meco .
    --------------------------------------------------------------------------------
    Processing Epoch 14: 100%|██████████| 910/910 [03:10<00:00,  4.77it/s, loss=1.978]
    --------------------------------------------------------------------------------
    SOURCE: He left soon after you did, and has not been in Moscow once since then.
    TARGET: È andato via poco dopo di te, e poi non è venuto neppure una volta a Mosca.
    PREDICTED: Subito vi ha lasciato presto , e non è tempo a Mosca .
    --------------------------------------------------------------------------------
    SOURCE: There was not one angry or anxious face among them.
    TARGET: Non c’era neppure un viso irritato o preoccupato.
    PREDICTED: Non vi era nessuno né né il volto .
    --------------------------------------------------------------------------------
    Processing Epoch 15: 100%|██████████| 910/910 [03:10<00:00,  4.77it/s, loss=1.963]
    --------------------------------------------------------------------------------
    SOURCE: I well remembered all; language, glance, and tone seemed at the moment vividly renewed.
    TARGET: Rammentati le sue parole, i suoi sguardi, la sua voce. Mi rammentavo tutto: il linguaggio, lo sguardo, l'accento mi tornarono alla mente.
    PREDICTED: Mi rammentai tutto la lingua , e il suo momento pareva che un momento fatale tutto con lui .
    --------------------------------------------------------------------------------
    SOURCE: 'Oh heavens! How many times!
    TARGET: — Oh Signore! quante volte!
    PREDICTED: — Oh come sono pochi !
    --------------------------------------------------------------------------------
    Processing Epoch 16: 100%|██████████| 910/910 [03:10<00:00,  4.77it/s, loss=1.784]
    --------------------------------------------------------------------------------
    SOURCE: 'If you want me to say what I think of it, I can only tell you that I don't believe in the tragedy.
    TARGET: — Se vuoi sapere il mio punto di vista, ti dirò che in questo non ci scorgo dramma. Ed ecco perché?
    PREDICTED: — Se mi volete dire quello che io dico , non posso dire , che in me non è certo alcun legame di esaltazione .
    --------------------------------------------------------------------------------
    SOURCE: 'WELL, KAPITONICH?' SAID SEREZHA, as on the day before his birthday he returned rosy and bright from a walk, and gave his overcoat to the tall old hall-porter, who looked smilingly down from his height at the little fellow.
    TARGET: — E allora, Kapitonyc? — disse Serëza, rosso in viso e allegro, tornando dalla passeggiata, alla vigilia del suo compleanno e dando il cappotto a pieghe al vecchio portiere che sorrideva al piccolo uomo dall’alto della sua statura. — Be’, è venuto oggi l’impiegato col braccio al collo?
    PREDICTED: — Su , via , ti prego , ’ ka al suo compleanno , è arrivato là di una carrozza magra , dal quale , sorridendo , s ’ era messo a guardare il portiere , aveva capito tutto il portiere , un sorriso di voci dal ragazzo .
    --------------------------------------------------------------------------------
    Processing Epoch 17: 100%|██████████| 910/910 [03:09<00:00,  4.79it/s, loss=1.701]
    --------------------------------------------------------------------------------
    SOURCE: 'I did not notice anything, Mama,' said Kitty, flushing up.
    TARGET: — No, non l’ho notato, maman — disse Kitty, avvampando.
    PREDICTED: — Non ho avuto nulla , mamma — disse Kitty , arrossendo .
    --------------------------------------------------------------------------------
    SOURCE: It was a lovely morning.
    TARGET: Era un mattino meraviglioso.
    PREDICTED: Era una bella mattina .
    --------------------------------------------------------------------------------

## Loss

Training Loss : 1.701

## Acknowledgments
This model is trained using repo listed below
* [language translation optimization](https://github.com/RInkalshah93/ERA-V2-Assignment_Rinkal-Shah/tree/815dae566694c71a8f30d1452bdcd4df64f4574c/S18_Assignment)