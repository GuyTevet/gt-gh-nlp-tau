id_1 = "305257206"
id_2 = "304946445"
email = "guy.tvt@gmail.com"

def get_details():
    if (not id_1) or (not id_2) or not (email):
        raise Exception("Missing submitters info")

    info = str.format("{}_{}      email: {}", id_1, id_2, email)

    return info