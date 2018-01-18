from datetime import datetime

from elasticsearch import Elasticsearch
from elasticsearch import helpers

es = Elasticsearch()
j = 0
str=["nulla","Liza Dunn","9.45","pop","aute","Pruitt Decker","9.47","pop","anim","Delia Merrill","3.98","bolly","esse","Sylvia Goff","6.28","rock","incididunt","Wiley Bender","5.51","jazz","irure","Cathy Glover","5.68","pop","quis","Montgomery Johns","6.89","pop","aliquip","Sophie Jacobson","9.18","jazz","non","Dotson Foster","5.4","rock","elit","Tabitha Gilbert","6.91","rock","laboris","Stephanie Gutierrez","2.09","pop","irure","Jennie Sargent","5.65","bolly","ipsum","Contreras Fuentes","7.45","bolly","officia","Dejesus Walton","5.38","rock","do","Monique Wise","8.06","rock","et","Wilda Burton","6.88","jazz","sint","Kimberly Munoz","5.65","pop","aliqua","Snow Bryan","3.35","bolly","est","Lowery Waller","8.65","jazz","reprehenderit","Jones Watkins","8","pop","minim","Wilkerson Luna","6.9","rock","magna","Payne Turner","5.3","bolly","veniam","Jaime Bennett","2.63","rock","cillum","Bette Garza","7.84","pop","laborum","Poole Santiago","4.28","jazz","aute","Muriel Greene","8.23","pop","labore","Fleming Carpenter","6.1","jazz","cupidatat","Mara Booker","6.72","bolly","ad","Loretta Clarke","3.99","jazz","laborum","Clara Christensen","5.41","jazz","adipisicing","Josie Rocha","3.94","pop","consectetur","Paige Ballard","9.86","pop","laborum","Leanne Hester","3.17","rock","velit","Britt Guy","5.17","rock","labore","Olson Morse","4.41","bolly","aute","Lynn James","4.49","rock","non","Cindy Herrera","5.34","rock","deserunt","Bruce Holloway","2.58","pop","esse","Kirsten Mooney","3.71","pop","minim","Naomi Velez","9.04","pop","nostrud","Rosanna Murray","5","jazz","non","Claudia Frost","4.01","rock","tempor","Farmer Tran","6.72","pop","nisi","Barrett Green","4.5","pop","id","Lopez Herring","2.07","jazz","proident","Cervantes Ball","4.59","rock","dolor","Gentry Callahan","7.86","jazz","deserunt","Beth Hansen","9.92","bolly","adipisicing","Sherman Shields","8.49","rock","esse","Gomez Woodard","8.81","pop","tempor","Baldwin Kline","6.57","pop"]
print str[1]
actions = []
while (j <=48*4 ):
	action = {
		"_index": "music",
		"_type": "data",
		"_id": j/4,
		"_source": {
			"song_name": str[j],
			"artist": str[j+1],
			"length": double(str[j+2]),
			"genre": str[j+3]

			}
		}
	actions.append(action)
	j += 4

helpers.bulk(es, actions)

