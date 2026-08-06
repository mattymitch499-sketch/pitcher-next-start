use crate::Result;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct PeopleResponse {
    people: Vec<Person>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Person {
    id: u32,
    pitch_hand: Hand,
}
#[derive(Deserialize)]
struct Hand {
    code: String,
}

pub fn normalize_pitcher_hands(json: &str) -> Result<HashMap<u32, char>> {
    let response: PeopleResponse = serde_json::from_str(json)?;
    Ok(response
        .people
        .into_iter()
        .filter_map(|person| {
            person
                .pitch_hand
                .code
                .chars()
                .next()
                .map(|hand| (person.id, hand))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalizes_handedness() {
        let hands = super::normalize_pitcher_hands(
            r#"{"people":[{"id":10,"pitchHand":{"code":"L"}},{"id":11,"pitchHand":{"code":"R"}}]}"#,
        )
        .unwrap();
        assert_eq!(hands[&10], 'L');
        assert_eq!(hands[&11], 'R');
    }
}
