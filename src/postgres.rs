use postgres::Client;

pub struct Postgres {
    client: Client,
}

impl Default for Postgres {
    fn default() -> Self {
        Self::new()
    }
}

impl Postgres {
    pub fn new() -> Postgres {
        let client = Client::connect("host=localhost user=postgres", postgres::NoTls).unwrap();
        Postgres { client }
    }
    pub fn create_table(&mut self) {
        self.client
            .execute(
                "CREATE TABLE IF NOT EXISTS tb_users (
                id              SERIAL PRIMARY KEY,
                name            VARCHAR NOT NULL,
                age             INT NOT NULL,
                email           VARCHAR NOT NULL
            )",
                &[],
            )
            .unwrap();
    }
    pub fn create(&mut self) {
        self.client
            .execute(
                "INSERT INTO tb_users (name, age, email) VALUES ($1, $2, $3)",
                &[&"John", &"20", &"test@gmail.com"],
            )
            .unwrap();
    }
    pub fn read(&mut self, id: i32) {
        for row in self
            .client
            .query("SELECT * FROM tb_users WHERE id=$1", &[&id])
            .unwrap()
        {
            let id: i32 = row.get(0);
            let name: &str = row.get(1);
            let age: i32 = row.get(2);
            let email: &str = row.get(3);
            println!("id: {}, name: {}, age: {}, email: {}", id, name, age, email);
        }
    }
    pub fn update(&mut self) {
        self.client
            .execute("UPDATE tb_users SET name=$1 WHERE id=$2", &[&"John", &"1"])
            .unwrap();
    }
    pub fn delete(&mut self) {
        self.client
            .execute("DELETE FROM tb_users WHERE id=$1", &[&"1"])
            .unwrap();
    }
}
