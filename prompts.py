#!/usr/bin/python3

import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.promtp import PromptTemplate
from langchain.output_parsers.regex import RegexParser

def few_shot_case(dataset = "tab_fact"):
  assert dataset in {'mmqa', 'tab_fact', 'wikiq'}
  if dataset == 'mmqa':
    examples = """Generate SQL given the question, table, passages, image captions to answer the question correctly.
If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar QA("map@").
If mapping to a new column still can not answer the question with valid SQL, turn to an end-to-end solution by a new grammar QA("ans@"). This grammar aims to solve all the rest of complex questions or tables or passages or image captions.

CREATE TABLE Dutch Ruppersberger (Electoral history)(
	row_id int,
	year int,
	office text,
	election text,
	filledcolumnname real,
	subject text,
	party text,
	votes text,
	% text,
	filledcolumnname_2 real,
	opponent text,
	party_2 text,
	votes_2 text,
	%_2 text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	office	election	filledcolumnname	subject	party	votes	%	filledcolumnname_2	opponent	party_2	votes_2	%_2
0	1994	baltimore county executive	general	nan	dutch ruppersberger	democratic	n/a	n/a	nan	n/a	n/a	n/a	n/a
1	1998	baltimore county executive	general	nan	dutch ruppersberger	democratic	166482	70.47	nan	john j. bishop	republican	69449	29.4
2	2002	none	general	nan	dutch ruppersberger	democratic	105718	54.16	nan	helen delich bentley	republican	88954	45.57
*/
Q: What year was Elizabeth Matory the opponent of Charles Albert Ruppersberger?
NeuralSQL: SELECT year FROM w WHERE opponent = 'elizabeth matory'


CREATE TABLE Virtual Console (Titles)(
	row_id int,
	system text,
	japan int,
	[[list of virtual console games for wii u (north america)|north  america]] real,
	pal region - europe real,
	pal region - australia real)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	system	japan	[[list of virtual console games for wii u (north america)|north  america]]	pal region - europe	pal region - australia
0	nes/famicom	148	94.0	89.0	89.0
1	super nes/super famicom	101	51.0	49.0	49.0
2	nintendo 64	22	21.0	21.0	21.0
*/
Q: Which system has a lower number for Japan of the virtual console systems: Game Boy Advance or the Japan-only console MSX?
NeuralSQL: SELECT system FROM w WHERE system IN ('game boy advance', 'msx (japan only)') ORDER BY japan LIMIT 1


CREATE TABLE 2018 Warrington Wolves season (Transfers | In)(
	row_id int,
	player text,
	signed from text,
	contract length text,
	announced text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	player	signed from	contract length	announced
0	sitaleki akauola	penrith panthers	p2y	2017-08-01 00:00:00
1	bryson goodwin	south sydney rabbitohs	p2y	2017-10-01 00:00:00
2	tyrone roberts	gold coast titans	p3y	2017-10-01 00:00:00
*/
CREATE TABLE Images(
	row_id int,
	gold coast titans text)
/*
All rows of the table:
SELECT * FROM w;
row_id	gold coast titans
0	a logo for the golden knights is painted on the beach.
*/
Q: What player was transferred from the team that has crossed swords on its logo to the Warrington Wolves in the 2018 season?
NeuralSQL: SELECT player FROM w WHERE QA("map@Has crossed swords on its logo?"; `signed from`) = 'yes'


CREATE TABLE 2013 Arizona Cardinals season (Regular season)(
	row_id int,
	week int,
	date text,
	opponent text,
	result text,
	record text,
	game site text,
	nfl.com recap text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	week	date	opponent	result	record	game site	nfl.com recap
0	1	september 8	at st. louis rams	l 24–27	0–1	edward jones dome	[http://www.nfl.com/gamecenter/2013090810/2013/reg1/cardinals@rams recap]
1	2	september 15	detroit lions	w 25–21	1–1	university of phoenix stadium	[http://www.nfl.com/gamecenter/2013091509/2013/reg2/lions@cardinals recap]
2	3	september 22	at new orleans saints	l 7–31	1–2	mercedes-benz superdome	[http://www.nfl.com/gamecenter/2013092207/2013/reg3/cardinals@saints recap]
*/
CREATE TABLE Passages(
	row_id int,
	candlestick park text)
/*
All rows of the table:
SELECT * FROM w;
row_id	candlestick park
0	candlestick park was an outdoor sports and entertainment stadium in the west coast of the united states, located in san francisco, in the bayview heights area. the stadium was originally the home of major league baseball's san francisco giants, who played there from 1960 until moving into pacific bell park (since renamed at&t park) in 2000. it was also the home field of the san francisco 49ers of the national football league from 1971 through 2013. the 49ers moved to levi's stadium in santa clara for the 2014 season.
*/
Q: In which year did the San Francisco 49ers move to their new stadium, which was the location that the Arizona Cardinals lost a 2013 regular season game by the score of 20 to 32?
NeuralSQL: SELECT QA("map@In which year did the San Francisco 49ers move to their new stadium?"; `game site`) FROM w WHERE opponent LIKE '%san francisco 49ers%' AND result = 'l 20–32'


CREATE TABLE PNC Park (Concerts)(
	row_id int,
	date text,
	artist text,
	opening act(s) text,
	tour / concert name text,
	attendance text,
	revenue text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	date	artist	opening act(s)	tour / concert name	attendance	revenue	notes
0	2003-08-06 00:00:00	bruce springsteen & the e street band	—	the rising tour	42301 / 48074	$3137575	none
1	2005-06-26 00:00:00	jimmy buffett	—	a salty piece of land tour	—	—	sonny landreth and jake shimabukuro were special guests http://www.buffettworld.com/archives/2005-a-salty-piece-of-land/6-26/
2	2005-09-28 00:00:00	the rolling stones	pearl jam	a bigger bang	—	—	none
*/
CREATE TABLE Passages(
	row_id int,
	can't stop won't stop (usher song) text,
	uptown girl text)
/*
All rows of the table:
SELECT * FROM w;
row_id	can't stop won't stop (usher song)	uptown girl
0	"can't stop won't stop" is a song recorded by american recording artist usher for his seventh studio album looking 4 myself (2012). written and produced by will "will.i.am" adams and keith harris, the song contains an interpolation of the bridge to billy joel's 1983 hit single "uptown girl". musically, "can't stop won't stop" is a eurodance and dance-pop song that incorporates elements of dubstep.	"uptown girl" is a song written and performed by american musician billy joel. it was released on 1983-9-29, on his ninth studio album an innocent man (1983). the lyrics describe a working-class "downtown man" attempting to woo a wealthy "uptown girl."
*/
Q: This song released on September 29, 1983 and inspired a hit song by Usher was written by who?
NeuralSQL: QA("ans@This song released on September 29, 1983 and inspired a hit song by Usher was written by who?"; Uptown Girl; Can't Stop Won't Stop (Usher song) )


CREATE TABLE 2000 DirecTV 500 (Top 10 results)(
	row_id int,
	pos int,
	grid int,
	car number (no.) int,
	driver text,
	team text,
	manufacturer text,
	laps completed (laps) int,
	points int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	pos	grid	car number (no.)	driver	team	manufacturer	laps completed (laps)	points
0	1	4	8	dale earnhardt, jr. (r)	dale earnhardt, inc.	chevrolet	334	185
1	2	37	99	jeff burton	roush racing	ford	334	175
2	3	14	18	bobby labonte	joe gibbs racing	pontiac	334	170
*/
CREATE TABLE Images(
	row_id int,
	dale earnhardt text)
/*
All rows of the table:
SELECT * FROM w;
row_id	dale earnhardt
0	a man wearing a number of neckties and a mustache.
*/
Q: The 2000 DirecTv 500 Top 10 Driver with 146 points has a person behind them holding what?
NeuralSQL: SELECT QA("map@what is the person behind holding?"; driver) FROM w WHERE points = 146


CREATE TABLE Oliver Mellor (Credits | Television)(
	row_id int,
	year text,
	title text,
	role text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	title	role	notes
0	2006	the royal	dr. guy fitzgerald	none
1	2006	hollyoaks: in the city	detective monroe	3 episodes
2	2006	doctor who	matt	episode "army of ghosts"
*/
CREATE TABLE Passages(
	row_id int,
	charlie stubbs (coronation street) text)
/*
All rows of the table:
SELECT * FROM w;
row_id	charlie stubbs (coronation street)
0	in 2005, charlie began a relationship with tracy barlow (kate ford). he convinced her to move in with him and later in february 2006, manipulated her into having her daughter amy (amber chadwick) move in with her parents. in turn, tracy began to manipulate charlie. she pretended to be pregnant and used the money he gave her for an abortion to buy expensive shoes and used her "grief" to have him allow amy to move back in. when shelley visited before her mother’s marriage to fred elliott (john savident), she and charlie had a 1-tni stand. she told tracy about their tni of passion, who accused her of lying. shelley later revealed that she was pregnant with charlie’s baby but didn’t allow charlie to have anything to do with the baby, and left. he and tracy briefly split but reconciled. charlie later began an affair with maria sutherland (samia smith), who was renting his flat. when david platt (jack p. shepherd) discovered the affair he tried to blackmail charlie, threatening to reveal the affair to tracy. charlie retaliated by trying to drown david in the bath. when tracy eventually found out about the affair, they split once more. tracy began to plot revenge against charlie and pretended to make amends with charlie. she pretended he was abusing her to the point of burning herself with an iron to make it look like charlie was responsible for her injuries. charlie eventually realized his partner was seeking revenge and when he was about to tell her their relationship was over, she insisted on performing a lap dance for him. she hit him round the head with a heavy ornament, and he later died in hospital. she claimed she’d killed him in self-defence but the court found her guilty and she was given a life sentence.
*/
Q: Oliver Mellor played Dr. Matt Carter on the TV show that had Tracy Barlow kill who?
NeuralSQL: QA("ans@Oliver Mellor played Dr. Matt Carter on the TV show that had Tracy Barlow kill who?"; SELECT title FROM w WHERE role = 'dr. matt carter'; Charlie Stubbs (Coronation Street))


CREATE TABLE Peter Egan (Filmography)(
	row_id int,
	year text,
	title text,
	role text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	title	role	notes
0	1971	1 brief su	bill denton	none
1	1971	elizabeth r	earl of southampton	episode: "sweet englands pride"
2	1973	the hireling	captain hugh cantrip	none
*/
CREATE TABLE Passages(
	row_id int,
	wampanoag text)
/*
All rows of the table:
SELECT * FROM w;
row_id	wampanoag
0	traditionally wampanoag people have been semi-sedentary, with seasonal movements between fixed sites in present-day southern new england. the men often traveled far north and south along the eastern seaboard for seasonal fishing expeditions, and sometimes stayed in those distant locations for weeks and months at a time. the women cultivated varieties of the "3 sisters" (the intercropping of maize, climbing beans, and squash) as the staples of their diet, supplemented by fish and game caught by the men. each community had authority over a well-defined territory from which the people derived their livelihood through a seasonal round of fishing, planting, harvesting, and hunting. because southern new england was thickly populated by indigenous peoples, hunting grounds had strictly defined boundaries.
*/
Q: corn beans and squash the three most important crops of the wampanoag were also known as
NeuralSQL: QA("ans@corn beans and squash the three most important crops of the wampanoag were also known as?"; Wampanoag)


CREATE TABLE 1980 in home video (Movie releases)(
	row_id int,
	u.s./canada release date text,
	title text,
	studio text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	u.s./canada release date	title	studio	notes
0	january 1	the muppet movie	magnetic video	betamax release laserdisc release   vhs release
1	march 4	20000 leagues under the sea	walt disney home entertainment	betamax release vhs release
2	march 4	the apple dumpling gang	walt disney home entertainment	betamax release vhs release
*/
CREATE TABLE Passages(
	row_id int,
	bert lahr text)
/*
All rows of the table:
SELECT * FROM w;
row_id	bert lahr
0	bert lahr ((1895-8-131967-12-4,p26410d)) was an american actor, particularly of stage and film, and comedian. lahr is known for his role as the cowardly lion, as well as his counterpart kansas farmworker zeke, in the wizard of oz (1939). he was well known for his explosive humor, but also adapted well to dramatic roles and his work in burlesque, vaudeville, and on broadway.
*/
Q: In the 1980 movie that was put out by the MGM/CBS Home Video studio, who played the part of the Cowardly Lion?
NeuralSQL: QA("ans@who played the part of the Cowardly Lion?"; SELECT title FROM w WHERE studio = 'mgm/cbs home video';  Bert Lahr)


CREATE TABLE List of newspapers in Italy (National daily newspapers)(
	row_id int,
	newspaper text,
	circulation text,
	headquarters text,
	est. int,
	political alignment text,
	nameplate text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	newspaper	circulation	headquarters	est.	political alignment	nameplate
0	corriere della sera	242684	milan	1876	centrism	200x200px
1	la repubblica	198835	rome	1976	social democracy	150x150px
2	la gazzetta dello sport	161796	milan	1896	—	200x200px
*/
CREATE TABLE Passages(
	row_id int,
	early middle ages text)
/*
All rows of the table:
SELECT * FROM w;
row_id	early middle ages
0	for almost p1000y, rome was the most politically important, richest and largest city in europe. around 100 ce, it had a population of about 450000, and declined to a mere 20000 during the early middle ages, reducing the sprawling city to groups of inhabited buildings interspersed among large areas of ruins and vegetation.
*/
CREATE TABLE Images(
	row_id int,
	rome text)
/*
All rows of the table:
SELECT * FROM w;
row_id	rome
0	a series of photographs showing a colorful scene.
*/
Q: In the city that was the center of imperial life in the roman empire in the early fifth century, the building in the top right has what at its top?
NeuralSQL: QA("ans@he building in the top right has what at its top?"; QA("ans@what is the city that was the center of imperial life in the roman empire in the early fifth century?"; Imperial fora))


CREATE TABLE International League (Current teams)(
	row_id int,
	division text,
	team text,
	founded int,
	mlb affiliation text,
	affiliated int,
	city text,
	stadium text,
	capacity int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	division	team	founded	mlb affiliation	affiliated	city	stadium	capacity
0	north	buffalo bisons	1985	toronto blue jays	2013	buffalo, new york	sahlen field	16600
1	north	lehigh valley ironpigs	2008	philadelphia phillies	2007	allentown, pennsylvania	coca-cola park	10100
2	north	pawtucket red sox	1973	boston red sox	1970	pawtucket, rhode island	mccoy stadium	10031
*/
CREATE TABLE Images(
	row_id int,
	columbus clippers text)
/*
All rows of the table:
SELECT * FROM w;
row_id	columbus clippers
0	a large blue and white clock on the side of a building.
*/
Q: Was the Team that has a ship in logo or Charlotte Knights, the one with earlier affiliation in Current teams of International League?
NeuralSQL: SELECT team FROM w WHERE team = 'charlotte knights' OR QA("map@Has a ship in logo or Charlotte Knights?"; team) = 'yes' ORDER BY founded LIMIT 1


CREATE TABLE Warren Burton (Filmography)(
	row_id int,
	year int,
	title text,
	role text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	title	role	notes
0	1976	baby blue marine	second serviceman	none
1	1977	chatterbox	tv reporter	none
2	1977	the world's greatest lover	ludwig	none
*/
CREATE TABLE Images(
	row_id int,
	green lantern (film) text)
/*
All rows of the table:
SELECT * FROM w;
row_id	green lantern (film)
0	a picture of a green and white costume and glasses.
*/
Q: How many people are on the poster for Green Lantern (film)?
NeuralSQL: QA("ans@How many people are on the poster for Green Lantern (film)?";  Green Lantern (film))


CREATE TABLE One Hour Photo (Accolades)(
	row_id int,
	award text,
	category text,
	recipients text,
	result real)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	award	category	recipients	result
0	critics' choice movie awards	best actor	robin williams	nan
1	dallas–fort worth film critics association	best actor	robin williams	nan
2	online film critics society	best actor	robin williams	nan
*/
CREATE TABLE Images(
	row_id int,
	saturn award text)
/*
All rows of the table:
SELECT * FROM w;
row_id	saturn award
0	a man in a suit and tie holding a glass.
*/
Q: What is he holding in Saturn Award?
NeuralSQL: QA("ans@What is he holding?"; Saturn Award)


CREATE TABLE 2013 Detroit Lions season (2013 Draft class)(
	row_id int,
	draft order - round int,
	draft order - choice int,
	draft order - overall int,
	player name text,
	position text,
	height text,
	weight text,
	college text,
	contract text,
	notes text,
	source text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	draft order - round	draft order - choice	draft order - overall	player name	position	height	weight	college	contract	notes	source
0	1	5	5	ezekiel ansah	defensive end	6ft 5 in	271lbs	byu	p5y /	none	[http://www.mlive.com/lions/index.ssf/2013-4/detroit_lions_select_ezekiel_a.html detroit lions select ezekiel ansah in first round of 2013 nfl draft] mlive.com, 2013-4-26
1	2	4	36	darius slay	defensive back	6ft 1 in	190lbs	mississippi state	p4y /	none	[http://www.mlive.com/lions/index.ssf/2013-4/detroit_lions_select_mississip.html detroit lions select mississippi state cb darius slay in second round of 2013 nfl draft] mlive.com, 2013-4-27
2	3	3	65	larry warford	offensive lineman	6ft 3 in	343lbs	kentucky	p4y /	none	[http://www.mlive.com/lions/index.ssf/2013-4/detroit_lions_fill_massive_nee.html detroit lions fill massive need with massive guard prospect larry warford] mlive.com, 2013-4-27
*/
CREATE TABLE Images(
	row_id int,
	south carolina gamecocks football text,
	seattle seahawks text)
/*
All rows of the table:
SELECT * FROM w;
row_id	south carolina gamecocks football	seattle seahawks
0	a group of people standing next to each other.	a large green and white bird with numbers.
*/
Q: What educational institution has a rooster on its logo and was the school listed in the 2013 Detroit Lions draft class for the defensive end player position?
NeuralSQL: QA("ans@which one has a rooster on his logo?"; SELECT college FROM w WHERE position='defensive end')


CREATE TABLE Melia Kreiling (Filmography |  Film roles)(
	row_id int,
	year int,
	title text,
	role text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	title	role	notes
0	2012	suspension of disbelief	juliette	none
1	2013	company of heroes	kestrel	direct-to-video film
2	2013	leopard	kara	none
*/
CREATE TABLE Passages(
	row_id int,
	list of marvel cinematic universe films text)
/*
All rows of the table:
SELECT * FROM w;
row_id	list of marvel cinematic universe films
0	the first film in the marvel cinematic universe was iron man (2008), which was distributed by paramount pictures. paramount also distributed iron man 2 (2010), thor (2011) and captain america: the first avenger (2011), while universal pictures distributed the incredible hulk (2008). walt disney studios motion pictures began distributing the films with the 2012 crossover film the avengers, which concluded phase 1 of the franchise. phase 2 includes iron man 3 (2013), thor: the dark world (2013), captain america: wi soldier (2014), guardians of the galaxy (2014), avengers: age of ultron (2015), and ant-man (2015).
*/
Q: What was Melia Kreiling's role in the film that is the next Marvel movie after 'Captain America the Winter Soldier'?
NeuralSQL: SELECT role FROM w WHERE title = QA("ans@which is the next  Marvel movie after 'Captain America the Winter Soldier'?"; List of Marvel Cinematic Universe films)


CREATE TABLE 2006 Grand Prix of Portland (Qualifying results)(
	row_id int,
	pos int,
	nat real,
	name text,
	team text,
	qual 1 text,
	qual 2 text,
	best text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	pos	nat	name	team	qual 1	qual 2	best
0	1	nan	bruno junqueira	newman/haas racing	59.576	57.631	57.631
1	2	nan	a. j. allmendinger	forsythe racing	58.378	57.639	57.639
2	3	nan	sébastien bourdais	newman/haas racing	58.464	57.646	57.646
*/
CREATE TABLE Passages(
	row_id int,
	jtg daugherty racing text)
/*
All rows of the table:
SELECT * FROM w;
row_id	jtg daugherty racing
0	jtg daugherty racing (formerly st motorsports and jtg racing) is an american professional stock car racing team that currently competes in the monster energy nascar cup series. the team is owned by former advertising executive tad geschickter and his wife jodi, along with current espn analyst brad daugherty. the team formerly had alliances with wood brothers racing, then michael waltrip racing, and currently has a technical alliance with richard childress racing. the team currently fields the no. 37 cottonelle chevrolet ss driven by roush development driver chris buescher and the no. 47 clorox/bush's/scott products chevrolet ss driven by a. j. allmendinger in the monster energy nascar cup series.
*/
Q: The driver of Nascar number 47 qualified for the 2006 Grand Prix of Portland for which team?
NeuralSQL: SELECT name FROM w WHERE team = QA("ans@which driver is number 47?"; JTG Daugherty Racing)


CREATE TABLE List of churches in Copenhagen ([[Amager]])(
	row_id int,
	name text,
	denomination text,
	year int,
	coordinates real,
	image text,
	refs real)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	name	denomination	year	coordinates	image	refs
0	all saints' church	church of denmark	1932	nan	150px	nan
1	dragør church	church of denmark	1885	nan	150px	nan
2	hans tausen's church	church of denmark	1924	nan	150px	nan
*/
CREATE TABLE Images(
	row_id int,
	all saints' church, copenhagen text,
	dragør church text,
	nathanael's church text,
	st. anne's church, copenhagen text,
	sundby church text)
/*
All rows of the table:
SELECT * FROM w;
row_id	all saints' church, copenhagen	dragør church	nathanael's church	st. anne's church, copenhagen	sundby church
0	 type of place of worship	 church of the holy trinity	 church of the holy trinity	 the building where the hotel is located	 a red brick church with a steeple and a flagpole in front of it.
*/
Q: Among Copenhagen churches on the "Amager" list, which have spires and are affiliated with the Church of Denmark denomination?
NeuralSQL: SELECT name FROM w WHERE denomination = 'church of denmark' AND QA("map@does it have spires?"; name) = 'yes'


CREATE TABLE Final Straw Tour (UK Tour (Leg III))(
	row_id int,
	date text,
	city text,
	country text,
	venue text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	date	city	country	venue
0	support acts: terra diablo & astrid	support acts: terra diablo & astrid	support acts: terra diablo & astrid	support acts: terra diablo & astrid
1	2004-3-2	newcastle	england	newcastle university
2	2004-3-3	liverpool	england	carling academy
*/
CREATE TABLE Images(
	row_id int,
	oxford text)
/*
All rows of the table:
SELECT * FROM w;
row_id	oxford
0	 a guide to the city of edinburgh
*/
Q: The final straw tour held leg 3 of the UK tour on March 13, 2004 in this city with how many views on the bottom?
NeuralSQL: SELECT QA("map@how many views on the bottom?"; city) FROM w WHERE date = '2004-3-13'"""
  elif dataset == 'tab_fact':
    examples = """Generate SQL given the statement and table to verify the statement correctly.
If statement-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar QA("map@").
If mapping to a new column still can not answer the statement with valid SQL, turn to an end-to-end solution by a new grammar QA("ans@"). This grammar aims to solve all the rest of complex statements or tables.

CREATE TABLE jason chambers(
	row_id int,
	res text,
	record text,
	opponent text,
	method text,
	event text,
	round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	res	record	opponent	method	event	round
0	win	18 - 5 - 2	dan new	submission (rear naked choke)	tfc - power fights	1
1	win	17 - 5 - 2	rene gonzalez	decision (split)	mainstream mma - cold war	n / a
2	loss	16 - 5 - 2	tristan yunker	submission ( armbar )	tfc 7 - total fight challenge 7	1
*/
Q: in mac - midwest absolute challenge , the player be defeat by dan spychalski in 1 round
NeuralSQL: SELECT (SELECT opponent, round FROM w WHERE event = "mac - midwest absolute challenge")=("dan spychalski", 1)


CREATE TABLE 1943 vfl season(
	row_id int,
	home team text,
	home team score text,
	away team text,
	away team score text,
	venue text,
	crowd int,
	date text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	home team	home team score	away team	away team score	venue	crowd	date
0	footscray	10.11 (71)	south melbourne	6.14 (50)	western oval	7500	1943-06-26 00:00:00
1	collingwood	10.21 (81)	melbourne	13.9 (87)	victoria park	5000	1943-06-26 00:00:00
2	carlton	15.16 (106)	fitzroy	9.13 (67)	princes park	12000	1943-06-26 00:00:00
*/
Q: western oval be the venue when the home team footscray score 10.11 (71)
NeuralSQL: SELECT (SELECT venue FROM w WHERE `home team`="footscray" AND `home team score`="10.11 (71)") = "western oval"


CREATE TABLE 2005 pba draft(
	row_id int,
	pick int,
	player text,
	country of origin text,
	pba team text,
	college text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	pick	player	country of origin	pba team	college
0	1	jay washington	united states	air21 express	eckerd
1	2	alex cabagnot	united states	sta lucia realtors	hawaii - hilo
2	3	dennis miranda	philippines	coca - cola tigers	feu
*/
Q: leo najorda be from philippine
NeuralSQL: SELECT (SELECT `country of origin` FROM w WHERE player = "leo najorda")="philippines"


CREATE TABLE none(
	row_id int,
	event text,
	long course / short course text,
	year set int,
	time text,
	meet text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	event	long course / short course	year set	time	meet
0	100 m freestyle	long course	2007	54.08	2007 fina world aquatic championships
1	200 m individual medley	long course	2011	2:11.23	2011 canadian world championship trials
2	4 x 100 m medley relay	long course	2010	3:38.14	2010 pan pacific championships
*/
Q: in 2009 the record be set in 7:51:80
NeuralSQL: SELECT (SELECT  time FROM w WHERE `year set` = 2009)="7:51.8"


CREATE TABLE turkish cup(
	row_id int,
	round text,
	clubs remaining int,
	clubs involved int,
	winners from previous round real,
	new entries this round real,
	leagues entering at this round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	round	clubs remaining	clubs involved	winners from previous round	new entries this round	leagues entering at this round
0	first round	156	86	nan	86.0	tff third league & turkish regional amateur league
1	second round	113	108	43.0	65.0	süper lig & tff first league & tff second league
2	third round	59	54	54.0	nan	none
*/
Q: during the 3rd round of the turkish cup , there be no new entry during that stage
NeuralSQL: SELECT (SELECT `new entries this round` FROM w WHERE round = 'third round') IS NULL


CREATE TABLE turkish cup(
	row_id int,
	round text,
	clubs remaining int,
	clubs involved int,
	winners from previous round real,
	new entries this round real,
	leagues entering at this round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	round	clubs remaining	clubs involved	winners from previous round	new entries this round	leagues entering at this round
0	first round	156	86	nan	86.0	tff third league & turkish regional amateur league
1	second round	113	108	43.0	65.0	süper ligs & tff first league & tff second league
2	third round	59	54	54.0	nan	none
*/
Q: süper lig be the most common league to win a round in the turkish cup
NeuralSQL: SELECT QA("ans@what is the most common league?"; `leagues entering at this round`) = 'süper ligs'


CREATE TABLE turkish cup(
	row_id int,
	round text,
	clubs remaining int,
	clubs involved int,
	winners from previous round real,
	new entries this round real,
	leagues entering at this round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	round	clubs remaining	clubs involved	winners from previous round	new entries this round	leagues entering at this round
0	first round	156	86	nan	86.0	tff third league & turkish regional amateur league
1	second round	113	108	43.0	65.0	süper lig & tff first league & tff second league
2	third round	59	54	54.0	nan	none
*/
Q: the lowest number of new entry conclude a round in the turkish cup be 5
NeuralSQL: SELECT (SELECT MIN(`new entries this round`) FROM w) = 5


CREATE TABLE cultural interest fraternities and sororities(
	row_id int,
	letters text,
	organization text,
	nickname text,
	founding time text,
	founding university text,
	type text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	letters	organization	nickname	founding time	founding university	type
0	αεπ	alpha epsilon pi 1	aepi	1913-11-07 00:00:00	new york university	fraternity
1	αεφ	alpha epsilon phi 2	aephi	1909-10-24 00:00:00	barnard college	sorority
2	σαεπ	sigma alpha epsilon pi 3	sigma	1998-10-01 00:00:00	university of california , davis	sorority
*/
Q: 4 of the cultural interest fraternity and sorority be fraternity while 3 be a sorority
NeuralSQL: SELECT (SELECT (SELECT COUNT(*) FROM w WHERE type = 'fraternity') = 4) AND (SELECT (SELECT COUNT(*) FROM w WHERE type = 'sorority') = 3)


CREATE TABLE british records in athletics(
	row_id int,
	event text,
	data text,
	athlete text,
	date text,
	place text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	event	data	athlete	date	place
0	5 km	t19:29	andi drake	1990-05-27 00:00:00	søfteland , norway
1	5 miles	32:38 +	ian mccombie	1985-03-23 00:00:00	york , united kingdom
2	10 km	40:17	chris maddocks	1989-04-30 00:00:00	burrator , united kingdom
*/
Q: there be 8 different event that take place within the united kingdom
NeuralSQL: SELECT (SELECT COUNT(place) FROM w WHERE QA("map@is it in united kingdom?"; place) = 'yes') = 8


CREATE TABLE jeev milkha singh(
	row_id int,
	tournament text,
	wins int,
	top - 10 int,
	top - 25 int,
	events int,
	cuts made int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	tournament	wins	top - 10	top - 25	events	cuts made
0	masters tournament	0	0	1	3	2
1	us open	0	0	0	4	3
2	the open championship	0	0	0	2	1
*/
Q: the number of cut made in the pga championship tournament be smaller than the number of event
NeuralSQL: SELECT (SELECT `cuts made` FROM w WHERE tournament = 'pga championship') < (SELECT events FROM w WHERE tournament = 'pga championship')


CREATE TABLE 2008 women 's british open(
	row_id int,
	place text,
	player text,
	country text,
	score int,
	to par int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par
0	1	juli inkster	united states	65	7
1	t2	momoko ueda	japan	66	6
2	t2	laura diaz	united states	66	6
*/
Q: the 3 player from japan have the same score
NeuralSQL: SELECT (SELECT COUNT(DISTINCT score) FROM w WHERE country = 'japan' GROUP BY score) = 1


CREATE TABLE espn sunday night football results (1987 - 2005)(
	row_id int,
	date text,
	visiting team text,
	final score text,
	host team text,
	stadium text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	date	visiting team	final score	host team	stadium
0	september 11	indianapolis colts	24 - 7	baltimore ravens	m&t bank stadium
1	september 18	kansas city chiefs	23 - 17	oakland raiders	mcafee coliseum
2	september 25	new york giants	23 - 45	san diego chargers	qualcomm stadium
*/
Q: the hosting team be the new york giant on new year even and the st louis ram on new year 's day
NeuralSQL: SELECT (SELECT (SELECT `host team` FROM w WHERE QA("map@is it new year even?"; date) = 'yes') = 'new york giant') AND (SELECT (SELECT `host team` FROM w WHERE QA("map@is it new year's day?"; date) = 'yes') = 'st louis ram')


CREATE TABLE 2008 women 's british open(
	row_id int,
	place text,
	player text,
	country text,
	score text,
	to par int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par
0	t1	yuri fudoh	japan	66 + 68 = 134	10
1	t1	jiyai shin	south korea	66 + 68 = 134	10
2	3	juli inkster	united states	65 + 70 = 135	9
*/
Q: kristie kerr , tie for 4th place , finish the round 1 stroke under lorena ochoa of mexico
NeuralSQL: SELECT (SELECT (SELECT QA("map@what is the derived score?"; score) FROM w WHERE player = 'cristie kerr') < (SELECT QA("map@what is the derived score?"; score) FROM w WHERE player = 'lorena ochoa' AND country = 'mexico')) AND (SELECT (SELECT place FROM w WHERE player = 'cristie kerr') = "t4")


CREATE TABLE connecticut public radio(
	row_id int,
	call sign text,
	frequency text,
	city of license text,
	facility id int,
	erp / power w int,
	height m ( ft ) real,
	class text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	call sign	frequency	city of license	facility id	erp / power w	height m ( ft )	class
0	waic	91.9 fm	springfield , ma	1749	230	nan	b1
1	wedw - fm	88.5 fm	stamford , ct	13619	2000	nan	a
2	wnpr	90.5 fm ( hd ) connecticut public radio	meriden , ct	13627	18500	nan	b
*/
Q: there be 3 station with a call sign number in the 90s
NeuralSQL: SELECT (SELECT COUNT(*) FROM w WHERE QA("map@is it in 90s?"; frequency) = 'yes' GROUP BY `call sign`) = 3


CREATE TABLE 2003 chicago white sox season(
	row_id int,
	date text,
	opponent text,
	score text,
	loss text,
	time text,
	att int,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	date	opponent	score	loss	time	att	record
0	august 1	mariners	12 - 1	garcía (9 - 11)	2:52	39337	58 - 51
1	august 2	mariners	0 - 10	wright (0 - 5)	2:22	45719	58 - 52
2	august 3	mariners	2 - 8	buehrle (9 - 11)	2:57	45632	58 - 53
*/
Q: the 2003 chicago white sox game play on 26th august be longer than the game play on 24th august
NeuralSQL: SELECT (SELECT time FROM w WHERE date = 'august 26') > (SELECT time FROM w WHERE date = 'august 24')


CREATE TABLE 1987 masters tournament(
	row_id int,
	place text,
	player text,
	country text,
	score text,
	to par text,
	money text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par	money
0	t1	larry mize	united states	70 + 72 + 72 + 71 = 285	-3	playoff
1	t1	bernhard langer	spain	73 + 71 + 70 + 71 = 285	-3	playoff
2	t1	greg norman	australia	73 + 74 + 66 + 72 = 285	-3	playoff
*/
Q: bernhard m. langer have more point than roger maltbie during the 1987 master tournament
NeuralSQL: SELECT (SELECT QA("map@what is the total score?"; score) FROM w WHERE player = 'bernhard langer') > (SELECT QA("map@what is the total score?"; score) FROM w WHERE player = 'roger maltbie')


CREATE TABLE 1987 masters tournament(
	row_id int,
	place text,
	player text,
	country text,
	score text,
	to par text,
	money text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par	money
0	t1	larry mize	united states	70 + 72 + 72 + 71 = 285	-3	playoff
1	t1	seve ballesteros	spain	73 + 71 + 70 + 71 = 285	-3	playoff
2	t1	greg norman	australia	73 + 74 + 66 + 72 = 285	-3	playoff
*/
Q: most of the people who play for the 1987 master tournament be spanish
NeuralSQL: SELECT (SELECT(SELECT COUNT(*) FROM w WHERE country = 'spain') / (SELECT COUNT(*) FROM w)) > 0.5


CREATE TABLE 1976 world junior figure skating championships(
	row_id int,
	rank int,
	name text,
	nation text,
	points real,
	places int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	rank	name	nation	points	places
0	1	sherri baier / robin cowan	canada	128.39	9
1	2	lorene mitchell / donald mitchell	united states	124.94	16
2	3	elizabeth cain / peter cain	australia	116.67	33
*/
Q: 2 of the 7 top - ranked figure skate team be from france
NeuralSQL: SELECT (SELECT (SELECT COUNT(*) FROM w) = 7) AND (SELECT (SELECT COUNT(*) FROM w WHERE nation = 'france') = 2)"""
  elif dataset == 'wikiq':
    examples = """Generate SQL given the question and table to answer the question correctly.
If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar QA("map@").
If mapping to a new column still can not answer the question with valid SQL, turn to an end-to-end solution by a new grammar QA("ans@"). This grammar aims to solve all the rest of complex questions or tables.

CREATE TABLE Fabrice Santoro(
	row_id int,
	name text,
	2001 text,
	2002 text,
	2003 text,
	2004 text,
	2005 text,
	2006 text,
	2007 text,
	2008 text,
	2009 text,
	2010 text,
	career\nsr text,
	career\nwin-loss text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	name	2001	2002	2003	2004	2005	2006	2007	2008	2009	2010	career\nsr	career\nwin-loss
0	australian open	2r	1r	3r	2r	1r	qf	3r	2r	3r	1r	0 / 18	22–18
1	french open	4r	2r	2r	3r	1r	1r	1r	2r	1r	a	0 / 20	17–20
2	wimbledon	3r	2r	2r	2r	2r	2r	2r	1r	2r	a	0 / 14	11–14
*/
Q: did he win more at the australian open or indian wells?
NeuralSQL: SELECT name FROM w WHERE name IN ('australian open', 'indian wells') ORDER BY QA("map@how many wins?"; `career\nwin-loss`) DESC LIMIT 1


CREATE TABLE 2007 New Orleans Saints season(
	row_id int,
	week int,
	date text,
	opponent text,
	time text,
	game site text,
	tv text,
	result/score text,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	week	date	opponent	time	game site	tv	result/score	record
0	1	2007-9-6	indianapolis colts	t20:30 edt	rca dome	nbc	l 41 – 10	0–1
1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	raymond james stadium	fox	l 31 – 14	0–2
2	3	2007-9-24	tennessee titans	t20:30 edt	louisiana superdome	espn	l 31 – 14	0–3
*/
Q: what number of games were lost at home?
NeuralSQL: SELECT COUNT(*) FROM w WHERE QA("map@is it a loss?"; `result/score`) = 'yes' AND QA("map@is it the home court of New Orleans Saints?"; `game site`) = 'yes'


CREATE TABLE 2007 New Orleans Saints season(
	row_id int,
	week int,
	date text,
	opponent text,
	time text,
	game site text,
	tv text,
	result/score text,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	week	date	opponent	time	game site	tv	result/score	record
0	1	2007-9-6	indianapolis colts	t20:30 edt	away	nbc	loss	0–1
1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	home	fox	win	1-1
2	3	2007-9-24	tennessee titans	t20:30 edt	away	espn	loss	1-2
*/
Q: what number of games were lost at home?
NeuralSQL: SELECT COUNT(*) FROM w WHERE `result/score` = 'loss' AND `game site` = 'home'


CREATE TABLE Demographics of Alaska(
	row_id int,
	by race text,
	white text,
	black text,
	aian* text,
	asian text,
	nhpi* text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	by race	white	black	aian*	asian	nhpi*
0	2000 (total population)	75.43%	4.46%	19.06%	5.24%	0.88%
1	2000 (hispanic only)	3.42%	0.33%	0.45%	0.16%	0.06%
2	2005 (total population)	74.71%	4.72%	18.77%	5.9%	0.88%
*/
Q: which hispanic population had the greatest growth from 2000 to 2005?
NeuralSQL: QA("ans@which race had the greatest value?"; SELECT white, black,  `aian*`, asian, `nhpi*` FROM w WHERE `by race` = 'growth 2000–5 (hispanic only)')


CREATE TABLE Highest mountain peaks of California(
	row_id int,
	rank int,
	mountain peak text,
	mountain range text,
	elevation text,
	prominence text,
	isolation text,
	location text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	rank	mountain peak	mountain range	elevation	prominence	isolation	location
0	1	mount whitney	sierra nevada	14505 ft; 4421 m	10080 ft; 3072 m	1646 mi; 2649 km	36°34′43″n 118°17′31″w﻿ / ﻿36.5786°n 118.292°w
1	2	mount williamson	sierra nevada	14379 ft; 4383 m	1677 ft; 511 m	5.4 mi; 8.7 km	36°39′21″n 118°18′40″w﻿ / ﻿36.6559°n 118.3111°w
2	3	white mountain peak	white mountains	14252 ft; 4344 m	7196 ft; 2193 m	67 mi; 109 km	37°38′3″n 118°15′21″w﻿ / ﻿37.6341°n 118.2557°w
*/
Q: which mountain peak has a prominence more than 10,000 ft?
NeuralSQL: SELECT `mountain peak` FROM w WHERE QA("map@prominence in ft?"; prominence) > 10000


CREATE TABLE Daegu FC(
	row_id int,
	season int,
	division int,
	tms. int,
	pos. int,
	fa cup text,
	afc cl text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	season	division	tms.	pos.	fa cup	afc cl
0	2003	1	12	11	quarter final	none
1	2004	1	13	10	round of 32	none
2	2005	1	13	8	quarter final	none
*/
Q: how far did they make it in the fa cup after 2009?
NeuralSQL: QA("ans@how far did they make?"; SELECT `fa cup` FROM w WHERE season > 2009)


CREATE TABLE Electricity in Sri Lanka(
	row_id int,
	filledcolumnname text,
	2005 int,
	2006 int,
	2007 int,
	2008 int,
	2009 int,
	2010 int,
	2011 int,
	2012 int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	filledcolumnname	2005	2006	2007	2008	2009	2010	2011	2012
0	hydro power	1293	1316	1326	1357	1379	1382	1401	1584
1	thermal	1155	1155	1155	1285	1290	1390	1690	1638
2	other renewables	3	3	3	3	15	45	50	90
*/
Q: did the hydro power increase or decrease from 2010 to 2012?
NeuralSQL: SELECT CASE WHEN (SELECT `2010` FROM w WHERE filledcolumnname = 'hydro power') < (SELECT `2012` FROM w WHERE filledcolumnname = 'hydro power') THEN 'increase' ELSE 'decrease' END


CREATE TABLE List of political parties in Japan(
	row_id int,
	party text,
	diet representation\nrepresentatives int,
	diet representation\ncouncillors int,
	party leader(s) text,
	comments text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	party	diet representation\nrepresentatives	diet representation\ncouncillors	party leader(s)	comments
0	your party (yp); minna no tō みんなの党; ("everybody's party")	18	18	yoshimi watanabe reps.	conservative liberalism, neoliberalism, economic liberalism, libertarianism, anti-nuclear
1	japanese communist party (jcp); nihon kyōsan-tō 日本共産党	8	11	kazuo shii reps.	the japanese communist party is japan's oldest party. it was formed in 1922 as an underground organization in the empire of japan, but was legalized after world war ii during the occupation. it used to be a communist party, but the party has past_ref shifted to a socialist party.
2	people's life party (plp); seikatsu no tō 生活の党	7	2	ichirō ozawa reps.	life party was founded by ichirō ozawa and 14 other diet members who were in the 2022-7-4 party of japan after a leadership dispute between ozawa and yukiko kada.
*/
Q: what party is listed previous to the new renaissance party?
NeuralSQL: SELECT QA("map@what is party name?"; party) FROM w WHERE row_id = (SELECT row_id FROM w WHERE QA("map@what is party name?"; party) = 'new renaissance party') - 1


CREATE TABLE FC Seoul in Asian football(
	row_id int,
	# int,
	season int,
	competition text,
	date text,
	round text,
	opponent text,
	h / a text,
	result text,
	scorer (s) text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	#	season	competition	date	round	opponent	h / a	result	scorer (s)
0	35	2011	afc; champions league	2011-03-02 00:00:00	group stage	al-ain	a	1–0	s : dejan damjanović
1	36	2011	afc; champions league	2011-03-15 00:00:00	group stage	hangzhou greentown	h	3–0	s : dejan damjanović, ou kyoung-jun, mauricio molina
2	37	2011	afc; champions league	2011-04-06 00:00:00	group stage	nagoya grampus	a	1–1	s : choi hyun-tae; n : kensuke nagai
*/
Q: how many consecutive games did dejan damjanovic score a goal in during the 2013 season?
NeuralSQL: QA("ans@how many consecutive games did dejan damjanovic score a goal?"; SELECT `scorer (s)` FROM w WHERE season = 2013)


CREATE TABLE Electoral district of Lachlan(
	row_id int,
	member text,
	party text,
	term text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	member	party	term
0	john ryan	none	1859–1864
1	james martin	none	1864–1869
2	james watson	none	1869–1880
*/
Q: of the members of the third incarnation of the lachlan, who served the longest?
NeuralSQL: SELECT member FROM w ORDER BY QA("map@how long does it last?"; term) DESC LIMIT 1


CREATE TABLE Portugal in the Eurovision Song Contest 1979(
	row_id int,
	draw int,
	artist text,
	song text,
	points int,
	place text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	draw	artist	song	points	place
0	1	gonzaga coutinho	"tema para um homem só"	102	5th
1	2	pedro osório s.a.r.l.	"uma canção comercial"	123	3rd
2	3	concha	"qualquer dia, quem diria"	78	6th
*/
Q: who was the last draw?
NeuralSQL: SELECT `artist` FROM w ORDER by `draw` desc LIMIT 1


CREATE TABLE GER Class N31(
	row_id int,
	year int,
	order text,
	quantity int,
	ger nos. text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	order	quantity	ger nos.
0	1893	n31	1	999
1	1893	h33	10	979
2	1894	l33	10	989
*/
Q: which had more ger numbers, 1898 or 1893?
NeuralSQL: SELECT `year` FROM w WHERE `year` IN ( '1898' , '1893' ) GROUP by `year` ORDER by SUM (`ger nos.`) desc LIMIT 1


CREATE TABLE List of spans(
	row_id int,
	tramway text,
	country text,
	city text,
	height of pylons text,
	span width,\nleaning straight line text,
	span width,\nhorizontal measurement text,
	height of cable over ground text,
	year of inauguration text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	tramway	country	city	height of pylons	span width,\nleaning straight line	span width,\nhorizontal measurement	height of cable over ground	year of inauguration	notes
0	peak 2 peak gondola	canada	whistler	65m	3024 m	3019 m	436 m	2008	3s aerial tramway constructed by doppelmayr
1	hut of regensburg material transport aerial railway	austria	falbeson	?	?	?	430 m	?	none
2	vanoise express	france	vanoise	none	1850 m	1800 m	380 m	2003	none
*/
Q: was the sandia peak tramway innagurate before or after the 3s aerial tramway?
NeuralSQL: SELECT ( SELECT `year of inauguration` FROM w WHERE `tramway` = 'sandia peak tramway' ) < ( SELECT `year of inauguration` FROM w WHERE `tramway` = '3s aerial tramway' )


CREATE TABLE Płock Governorate(
	row_id int,
	language text,
	number int,
	percentage (%) text,
	males int,
	females int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	language	number	percentage (%)	males	females
0	polish	447685	80.86	216794	230891
1	yiddish	51215	9.25	24538	26677
2	german	35931	6.49	17409	18522
*/
Q: how many male and female german speakers are there?
NeuralSQL: SELECT `males` + `females` FROM w WHERE `language` = 'german'"""
  else:
    raise NotImplementedError
  return examples

def get_binder_template(tokenizer,
                        prompt_style = 'create_table_select_3_full_table',
                        generate_type = 'answer',
                        title: str,
                        table: pd.DataFrame):
  assert prompt_style in {'create_table_select_3_full_table',
                          'create_table_select_full_table',
                          'create_table_select_3',
                          'create_table',
                          'create_table_select_3_full_table_w_all_passage_image',
                          'create_table_select_3_full_table_w_gold_passage_image',
                          'no_table'}
  assert generate_type in {'answer', 'nsql', 'sql', 'npython', 'python'}
  system_message = "I will give you some x-y examples followed by a x, you need to give me the y, and no other content."
  user_message = ""
  if generate_type == 'answer':
    user_message += """\n-- Answer the question based on the given table below.\n\n"""
  elif generate_type == 'nsql':
    user_message += """\n-- Parse the question into NeuralSQL based on the given table below.\n\n"""
  elif generate_type == 'sql':
    user_message += """\n-- Parse the question into SQL based on the given table below.\n\n"""
  elif generate_type == 'npython':
    user_message += """\n-- Parse the question into NeuralPython based on the given table below.\n\n"""
  elif generate_type == 'python':
    user_message += """\n-- Parse the question into Python based on the given table below.\n\n"""
  else:
    raise NotImplementedError
  if prompt_style != 'no_table':
    # create table sql
    user_message += "CREATE TABLE %s(" % title
    for idx, header in enumerate(df.columns):
      column_type = {'int64':'int',
                     'float64':'real',
                     'datetime64':'datetime',
                     'text':'text'}[df[header].dtype]
      if idx != len(df.columns) - 1:
        user_message += "\t%s %s,\n" % (header, column_type)
      else:
        user_message += "\t%s %s)\n" % (header, column_type)
  # 
  messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': user_message}]
