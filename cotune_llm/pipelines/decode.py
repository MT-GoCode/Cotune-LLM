from cotune_llm.components.decoder import Decoder


def decode_workflow():
    ex = [
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 133401, age is 7, gender is 2",
        "user_id is 270403, age is 8, gender is 2",
        "user_id is 139353, age is 8, gender is 3",
        "user_id is 195306, age is 5, gender is 2",
    ]
    print(Decoder(ex).GReaT_auto_decode())
